import embeddings
from cupy_utils import *
import argparse
import collections
import numpy as np
import sys
import pickle
import time
import torch
import random
from new_dssm import DssmTrainer
import json
import copy

def debug_neg_sampling_record(src_w2negs, src_ind2word, trg_ind2word, train_set):
    # 保存hard neg sample结果用于debug
    neg_result = []
    correct_mapping = collections.defaultdict(set)
    for s, t in train_set:
      correct_mapping[s].add(t)    

    for src in src_w2negs:
      src_word = src_ind2word[src]
      neg_trg_word_list = src_w2negs[src]
      neg_trg_word_list = sorted(neg_trg_word_list, key=lambda x:x[1], reverse=True)

      neg_trg_word_list = [trg_ind2word[_[0]] + '(' + "%.5f" % _[1] + ')' for _ in neg_trg_word_list]
      neg_result.append({
        'src_word': src_word,
        'gold_tgt': [trg_ind2word[_] for _ in correct_mapping[src]],
        'hard_neg_sample_words': ','.join(neg_trg_word_list[:100])
      })
    
    with open('orig_neg_select_v4.json', 'w') as f:
      json.dump(neg_result, f, indent=2, ensure_ascii=False)

def debug_sample_wise_neg(sample2tgtsneg, sample2srcsneg, src_ind2word, trg_ind2word, train_set):
    # 保存hard neg sample结果用于debug
    neg_result = []
    for s, t in train_set:
      if s in sample2tgtsneg:
        neg_trg_word_list = sample2tgtsneg[s]
      else:
        neg_trg_word_list = sample2tgtsneg[(s, t)]
      neg_trg_word_list = sorted(neg_trg_word_list, key=lambda x:x[1], reverse=True)
      neg_trg_word_list = [trg_ind2word[_[0]] + '(' + "%.5f" % _[1] + ')' for _ in neg_trg_word_list]
      neg_src_word_list = []
      if sample2srcsneg is not None:
        neg_src_word_list = sample2srcsneg[(s, t)] 
        neg_src_word_list = sorted(neg_src_word_list, key=lambda x:x[1], reverse=True)
        neg_src_word_list = [src_ind2word[_[0]] + '(' + "%.5f" % _[1] + ')' for _ in neg_src_word_list]

      neg_result.append({
        'src': src_ind2word[s],
        'tgt': trg_ind2word[t],
        'hard_neg_tgt_words': ','.join(neg_trg_word_list[:100]),
        'hard_neg_src_words': ','.join(neg_src_word_list[:100])
      })
    
    with open('orig_neg_select_new_pipeline.json', 'w') as f:
      json.dump(neg_result, f, indent=2, ensure_ascii=False)
    exit(0)


def debug_monolingual_nns(xw, zw, src_ind2word, trg_ind2word):
    x_cos_sim = xw.dot(xw.T)
    x_nn = (-x_cos_sim).argsort(axis=1)
    tmp_sim = - x_cos_sim
    tmp_sim.sort(axis=1)
    tmp_sim = - tmp_sim

    src_word2nn = {}
    for i in range(x_cos_sim.shape[0]):
      nn_word = [src_ind2word[_] for _ in x_nn[i].tolist()]
      nn_score = tmp_sim[i].tolist()
      src_word2nn[src_ind2word[i]] = list(zip(nn_word, nn_score))[:100]

    z_cos_sim = zw.dot(zw.T)
    z_nn = (-z_cos_sim).argsort(axis=1)
    tmp_sim = - z_cos_sim
    tmp_sim.sort(axis=1)
    tmp_sim = - tmp_sim

    tgt_word2nn = {}
    for i in range(z_cos_sim.shape[0]):
      nn_word = [trg_ind2word[_] for _ in z_nn[i].tolist()]
      nn_score = tmp_sim[i].tolist()
      tgt_word2nn[trg_ind2word[i]] = list(zip(nn_word, nn_score))[:100] 

    with open('src_word2nn.json', 'w') as f:
      json.dump(src_word2nn, f, indent=2, ensure_ascii=False) 
    with open('tgt_word2nn.json', 'w') as f:
      json.dump(tgt_word2nn, f, indent=2, ensure_ascii=False)

def get_NN(src, X, Z, num_NN, cuda = False, batch_size = 100, return_scores = True):
  # get Z to the GPU once in the beginning (it can be big, seems like a waste to copy it again for every batch)
  if cuda:
    if not supports_cupy():
      print("Error: Install CuPy for CUDA support", file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    z = xp.asarray(Z)
  else: # not cuda
    z = Z
  X = copy.deepcopy(X)
  Z = copy.deepcopy(Z)
  #embeddings.normalize(X, ['unit', 'center', 'unit'])
  #embeddings.normalize(Z, ['unit', 'center', 'unit'])
 
  ret = {}

  # 按batch size计算NN
  print("Starting NN")
  start_time = time.time()
  for i in range(0, len(src), batch_size):

    j = min(i + batch_size, len(src))    
    x_batch_slice = X[src[i:j]]
    
    # get the x part to the GPU if needed
    if cuda:
     x_batch_slice = xp.asarray(x_batch_slice)
    
    similarities = x_batch_slice.dot(z.T)

    # 加符号，这样就是从大到小sort了
    nn = (-similarities).argsort(axis=1)

    for k in range(j-i):
      ind = nn[k,0:num_NN].tolist()
      if return_scores:             
        sim = similarities[k,ind].tolist()
        ret[src[i+k]] = list(zip(ind,sim))
      else:
        ret[src[i+k]] = ind
  print("Time taken " + str(time.time() - start_time))
  return(ret)   

def generate_negative_examples_v1(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True): 
  l = len(positive_examples)
  src_word2neg_words = collections.defaultdict(list)

  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]

  pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
  pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
  
  # src_word_ind -> tgt_word_ind 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)
  #correct_mapping = dict(zip(pos_src_word_indexes, pos_tar_word_indexes))

  # nns maps src word indexes to a list of tuples (tar_index, similarity)
  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  for src_ind in nns:
    # 将src对应的多个tgt word的topk nearest neighbor word index构建集合
    tgt_nn_wi = nns[src_ind][:top_k]
    # 去掉groundtruth与重复单词
    candidate_neg_wi2s = dict()
    for wi, s in tgt_nn_wi:
      if wi not in candidate_neg_wi2s and wi not in correct_mapping[src_ind]:
        candidate_neg_wi2s[wi] = s
    
    candidate_neg_wi = sorted(list(candidate_neg_wi2s.items()), key=lambda x:x[1], reverse=True)
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      src_word2neg_words[src_ind].append((neg_wi, neg_s))

  return src_word2neg_words, None

# method can be "random", "hard" or "mix"
def generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0): 
  l = len(positive_examples)
  src_word2neg_words = collections.defaultdict(list)

  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]
  
  pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
  pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  # 找自己语言空间中的最近邻作为hard neg example
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)

  # 为每个pos_src word挑num_neg_per_pos个负例
  for src_ind in correct_mapping:
    # 将src对应的多个tgt word的topk nearest neighbor word index构建集合
    tgt_nn_wi = [_ for tgt_wi in correct_mapping[src_ind] for _ in tgt_nns[tgt_wi][:top_k] if _[1] > hard_threshold]
    tgt_nn_wi = sorted(tgt_nn_wi, key=lambda x:x[1])
    # 去掉groundtruth与重复单词
    candidate_neg_wi2s = dict()
    for wi, s in tgt_nn_wi:
      if wi not in candidate_neg_wi2s and wi not in correct_mapping[src_ind]:
        candidate_neg_wi2s[wi] = s

    candidate_neg_wi = sorted(list(candidate_neg_wi2s.items()), key=lambda x:x[1], reverse=True)
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      src_word2neg_words[src_ind].append((neg_wi, neg_s))

  return src_word2neg_words, None

def generate_negative_samplewise_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0):
  l = len(positive_examples)
  pos_pair2neg_words = collections.defaultdict(list)

  pos_src_word_indexes = [src_w2ind[e[0]] for e in positive_examples]
  pos_tar_word_indexes = [tar_w2ind[e[1]] for e in positive_examples]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例
  # 为每个(pos_src, pos_tgt) pair挑num_neg_per_pos个负例
  for src_ind, tgt_ind in zip(pos_src_word_indexes, pos_tar_word_indexes):
    tgt_nn_wi = tgt_nns[tgt_ind][:top_k]
    tgt_nn_wi_filter_lowscore = [(w, s) for w, s in tgt_nn_wi if s > hard_threshold]
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in tgt_nn_wi_filter_lowscore if w not in correct_mapping[src_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    pos_pair2neg_words[(src_ind, tgt_ind)] = hard_neg_inds_sample

  return pos_pair2neg_words, None

# method can be "random", "hard" or "mix"
def generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0): 
  l = len(positive_examples)
  src_word2neg_words = collections.defaultdict(list)

  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]
  
  pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
  pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例

  # 为每个pos_src word挑num_neg_per_pos个负例
  for src_ind in correct_mapping:
    # 将src对应的多个tgt word的topk nearest neighbor word index构建集合
    tgt_nn_wi = [_ for tgt_wi in correct_mapping[src_ind] for _ in tgt_nns[tgt_wi][:top_k]]
    tgt_nn_wi = sorted(tgt_nn_wi, key=lambda x:x[1])
    # 去掉groundtruth与重复单词
    candidate_neg_wi2s = dict()
    for wi, s in tgt_nn_wi:
      if wi not in candidate_neg_wi2s and wi not in correct_mapping[src_ind]:
        candidate_neg_wi2s[wi] = s

    candidate_neg_wi = sorted(list(candidate_neg_wi2s.items()), key=lambda x:x[1], reverse=True)
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      src_word2neg_words[src_ind].append((neg_wi, neg_s))

  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 取训练集中出现的pos_src的最近邻pos_src_nn1, pos_src_nn2, ...
  # 将 pos_tgt1, pos_tgt2, ... 作为pos_src_nn1, pos_src_nn2, ...的hard负例

  for src_ind in correct_mapping:
    # 拿到src的topk nearest neighbor word index集合
    src_nn_wi = [_[0] for _ in src_nns[src_ind] if _[0] in correct_mapping]
    tgt_sets = correct_mapping[src_ind]
    
    # 去掉groundtruth
    for src_nn_ind in src_nn_wi:
      candidate_neg_wi = list(set(tgt_sets) - correct_mapping[src_nn_ind])
      src_word2neg_words[src_nn_ind].extend([(neg_wi, 0) for neg_wi in candidate_neg_wi])

  return src_word2neg_words, None

def generate_negative_samplewise_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0):
  l = len(positive_examples)
  pos_pair2neg_words = collections.defaultdict(list)

  pos_src_word_indexes = [src_w2ind[e[0]] for e in positive_examples]
  pos_tar_word_indexes = [tar_w2ind[e[1]] for e in positive_examples]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例
  # 为每个(pos_src, pos_tgt) pair挑num_neg_per_pos个负例
  neg_examples = []
  for src_ind, tgt_ind in zip(pos_src_word_indexes, pos_tar_word_indexes):
    tgt_nn_wi = tgt_nns[tgt_ind][:top_k]
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in tgt_nn_wi if w not in correct_mapping[src_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    pos_pair2neg_words[(src_ind, tgt_ind)] = hard_neg_inds_sample

  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 取训练集中出现的pos_src的最近邻pos_src_nn1, pos_src_nn2, ...
  # 将 pos_tgt1, pos_tgt2, ... 作为pos_src_nn1, pos_src_nn2, ...的hard负例

  for pos_src_ind in correct_mapping:
    # 拿到src的topk nearest neighbor word index集合
    src_nn_wi = [_[0] for _ in src_nns[pos_src_ind] if _[0] in correct_mapping]
    tgt_sets = correct_mapping[pos_src_ind]
    
    # 待添加负例的训练数据
    candidate_pos_pair = [(pos_src_nn, _) 
                          for pos_src_nn in src_nn_wi 
                          for _ in correct_mapping[pos_src_nn]]

    # 去掉groundtruth与重复负例
    for s, t in candidate_pos_pair:
      candidate_neg_wi = tgt_sets
      # 与 groundth 去重
      candidate_neg_wi = candidate_neg_wi - correct_mapping[s]
      # 与 已有负例 去重
      exist_neg_wi = [_[0] for _ in pos_pair2neg_words[(s, t)]]
      candidate_neg_wi = candidate_neg_wi - set(exist_neg_wi)
      
      # 加入负例
      candidate_neg_wi = [(_, 0) for _ in candidate_neg_wi]
      pos_pair2neg_words[(s, t)].extend(candidate_neg_wi)

  return pos_pair2neg_words, None


def generate_negative_examples_v4(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True):
  xp = get_array_module(x)
  sim_size = min(x.shape[0], z.shape[0])
  u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
  xsim = (u*s).dot(u.T)
  u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
  zsim = (u*s).dot(u.T)
  del u, s, vt
  xsim.sort(axis=1)
  zsim.sort(axis=1)
  normalize_method = ['unit', 'center', 'unit']
  embeddings.normalize(xsim, normalize_method)
  embeddings.normalize(zsim, normalize_method)
  src_word2neg_words_sim_distri, _ = generate_negative_examples_v1(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, xsim, zsim, top_k, num_neg_per_pos, hard_neg_random)
  del xsim, zsim
  src_word2neg_words_neg, _ = generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random)
  
  agg_src_word2_neg_words = {}
  mean_len_list = []
  for _src in src_word2neg_words_sim_distri:
    src_word2neg_words_sim_distri[_src] = [(_[0], 0) for _ in src_word2neg_words_sim_distri[_src]]
    src_word2neg_words_neg[_src] = [(_[0], 0) for _ in src_word2neg_words_neg[_src]]
    neg_set = list(set(src_word2neg_words_sim_distri[_src] + src_word2neg_words_neg[_src]))
    agg_src_word2_neg_words[_src] = neg_set
    mean_len_list.append(len(neg_set))
  print(sum(mean_len_list) / len(mean_len_list))
  return agg_src_word2_neg_words, None

def generate_negative_examples_v5(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True):
  xp = get_array_module(x)
  sim_size = min(x.shape[0], z.shape[0])
  u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
  xsim = (u*s).dot(u.T)
  u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
  zsim = (u*s).dot(u.T)
  del u, s, vt
  xsim.sort(axis=1)
  zsim.sort(axis=1)
  normalize_method = ['unit', 'center', 'unit']
  embeddings.normalize(xsim, normalize_method)
  embeddings.normalize(zsim, normalize_method)
  src_word2neg_words_sim_distri, _ = generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, xsim, zsim, top_k, num_neg_per_pos, hard_neg_random)
  del xsim, zsim
  src_word2neg_words_neg, _ = generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random)
  
  agg_src_word2_neg_words = {}
  mean_len_list = []
  for _src in src_word2neg_words_sim_distri:
    src_word2neg_words_sim_distri[_src] = [(_[0], 0) for _ in src_word2neg_words_sim_distri[_src]]
    src_word2neg_words_neg[_src] = [(_[0], 0) for _ in src_word2neg_words_neg[_src]]
    neg_set = list(set(src_word2neg_words_sim_distri[_src] + src_word2neg_words_neg[_src]))
    agg_src_word2_neg_words[_src] = neg_set
    mean_len_list.append(len(neg_set))
  print(sum(mean_len_list) / len(mean_len_list))
  return agg_src_word2_neg_words, None

def generate_negative_examples_v6(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True):
  l = len(positive_examples)
  pos_pair2neg_words = collections.defaultdict(list)

  pos_src_word_indexes = [src_w2ind[e[0]] for e in positive_examples]
  pos_tar_word_indexes = [tar_w2ind[e[1]] for e in positive_examples]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例

  # 为每个(pos_src, pos_tgt) pair挑num_neg_per_pos个负例
  neg_examples = []
  for src_ind, tgt_ind in zip(pos_src_word_indexes, pos_tar_word_indexes):
    tgt_nn_wi = tgt_nns[tgt_ind][:top_k]
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in tgt_nn_wi if w not in correct_mapping[src_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    pos_pair2neg_words[(src_ind, tgt_ind)] = hard_neg_inds_sample

  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 取训练集中出现的pos_src的最近邻pos_src_nn1, pos_src_nn2, ...
  # 将 pos_tgt1, pos_tgt2, ... 作为pos_src_nn1, pos_src_nn2, ...的hard负例

  for pos_src_ind in correct_mapping:
    # 拿到src的topk nearest neighbor word index集合
    src_nn_wi = [_[0] for _ in src_nns[pos_src_ind] if _[0] in correct_mapping]
    tgt_sets = correct_mapping[pos_src_ind]
    
    # 待添加负例的训练数据
    candidate_pos_pair = [(pos_src_nn, _) 
                          for pos_src_nn in src_nn_wi 
                          for _ in correct_mapping[pos_src_nn]]

    # 去掉groundtruth与重复负例
    for s, t in candidate_pos_pair:
      candidate_neg_wi = tgt_sets
      # 与 groundth 去重
      candidate_neg_wi = candidate_neg_wi - correct_mapping[s]
      # 与 已有负例 去重
      exist_neg_wi = [_[0] for _ in pos_pair2neg_words[(s, t)]]
      candidate_neg_wi = candidate_neg_wi - set(exist_neg_wi)
      
      # 加入负例
      candidate_neg_wi = [(_, 0) for _ in candidate_neg_wi]
      pos_pair2neg_words[(s, t)].extend(candidate_neg_wi)

  return pos_pair2neg_words, None

def generate_negative_examples_v7(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, hard_neg_random=True):
  l = len(positive_examples)
  sample2neg_tgt = collections.defaultdict(list)
  sample2neg_src = collections.defaultdict(list)

  pos_src_word_indexes = [src_w2ind[e[0]] for e in positive_examples]
  pos_tar_word_indexes = [tar_w2ind[e[1]] for e in positive_examples]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  src2groundth = collections.defaultdict(set)
  tgt2groundth = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    src2groundth[s].add(t)
    tgt2groundth[t].add(s)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例

  # 为每个(pos_src, pos_tgt) pair挑num_neg_per_pos个负例
  for src_ind, tgt_ind in zip(pos_src_word_indexes, pos_tar_word_indexes):
    tgt_nn_wi = tgt_nns[tgt_ind][:top_k]
    src_nn_wi = src_nns[src_ind][:top_k]

    # src->tgt方向的负例
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in tgt_nn_wi if w not in src2groundth[src_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    sample2neg_tgt[(src_ind, tgt_ind)] = hard_neg_inds_sample

    # tgt->src方向的负例
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in src_nn_wi if w not in tgt2groundth[tgt_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    sample2neg_src[(src_ind, tgt_ind)] = hard_neg_inds_sample    

  return sample2neg_tgt, sample2neg_src


NEG_SAMPLING_METHOD = {
  'a' : generate_negative_examples_v2,
  'samplewise_a' : generate_negative_samplewise_examples_v2,
  'ab' : generate_negative_examples_v3,
  'samplewise_ab' : generate_negative_examples_v6,
  'abc' : generate_negative_examples_v4,
  'abd' : generate_negative_examples_v5,
  'samplewise': generate_negative_examples_v6,
  'bi_samplewise': generate_negative_examples_v7
}
