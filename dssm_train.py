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
from graph_utils import calc_csls_sim

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
        'hard_neg_tgt_words': ','.join(neg_trg_word_list),
        'hard_neg_src_words': ','.join(neg_src_word_list)
      })
    
    with open('orig_neg_select_new_pipeline_cos.json', 'w') as f:
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

def get_NN(src, X, Z, num_NN, cuda = False, batch_size = 100, return_scores = True, method="cos"):
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
  if method == "csls":
    torch_X = torch.tensor(X)
    torch_Z = torch.tensor(Z)
    sim_matrix = torch.matmul(torch_X, torch_Z.transpose(0,1))
    sim_src_topk, _ = torch.topk(sim_matrix, 11, dim=1)
    rt = torch.mean(sim_src_topk[:, 1:], dim=1)
    sim_tgt_topk, _ = torch.topk(sim_matrix.transpose(0,1), 11, dim=1)
    rs = torch.mean(sim_tgt_topk[:, 1:], dim=1)
    rs = xp.asarray(rs.numpy())
    rt = xp.asarray(rt.numpy())
    del torch_X, torch_Z

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
    
    if method == "cos":
      similarities = x_batch_slice.dot(z.T)
    else:
      similarities = x_batch_slice.dot(z.T) * 2 - rt[src[i:j], None] - rs     

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

# method can be "random", "hard" or "mix"
def generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"): 
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
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
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

def generate_negative_samplewise_examples_v2(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"):
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
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
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
def generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"): 
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

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
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

def generate_negative_samplewise_examples_v3(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"):
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

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
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

def generate_negative_examples_v7(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"):
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

  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
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

def generate_negative_examples_v8(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, hard_threshold=0, method="cos"):
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
  src_nns = get_NN(list(range(x.shape[0])), x, x, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  tgt_nns = get_NN(list(range(z.shape[0])), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
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


  sample2neg_tgt_b = collections.defaultdict(set)
  sample2neg_src_b = collections.defaultdict(set)
   
  for src_ind, tgt_ind in sample2neg_tgt:
    src_nn_wi_in_data = [wi for wi, si in src_nns[src_ind][:top_k] if wi in src2groundth]
    candidate_sample = [(wi, wi2t) for wi in src_nn_wi_in_data
                                   for wi2t in src2groundth[wi]]
    for s, t in candidate_sample:
      sample2neg_tgt_b[(s, t)].add(tgt_ind)

    tgt_nn_wi_in_data = [wi for wi, si in tgt_nns[tgt_ind][:top_k] if wi in tgt2groundth]
    candidate_sample = [(wi2s, wi) for wi in tgt_nn_wi_in_data
                                   for wi2s in tgt2groundth[wi]]    
    for s, t in candidate_sample:
      sample2neg_src_b[(s, t)].add(src_ind)
  
  # 合并两种方式的负例，去重
  for src_ind, tgt_ind in sample2neg_tgt:
    exist_neg_tgt = [_[0] for _ in sample2neg_tgt[(s, t)]]
    candidate_neg_tgt_b = sample2neg_tgt_b[(s, t)] - set(exist_neg_tgt)
    candidate_neg_tgt_b = [(w, 0.5) for w in candidate_neg_tgt_b]
    sample2neg_tgt[(s, t)].extend(candidate_neg_tgt_b)

    exist_neg_src = [_[0] for _ in sample2neg_src[(s, t)]]
    candidate_neg_src_b = sample2neg_src_b[(s, t)] - set(exist_neg_src)
    candidate_neg_src_b = [(w, 0.5) for w in candidate_neg_src_b]
    sample2neg_src[(s, t)].extend(candidate_neg_src_b)

  return sample2neg_tgt, sample2neg_src

NEG_SAMPLING_METHOD = {
  'a' : generate_negative_examples_v2,
  'samplewise_a' : generate_negative_samplewise_examples_v2,
  'ab' : generate_negative_examples_v3,
  'samplewise_ab' : generate_negative_samplewise_examples_v3,
  'bi_samplewise_a': generate_negative_examples_v7,
  'bi_samplewise_ab' : generate_negative_examples_v8
}


def whitening_transformation_v1(embedding):
  xp = get_array_module(embedding)
  miu = xp.mean(embedding, 0)
  _embedding = embedding - miu
  zigma = xp.zeros(shape=(embedding.shape[1], embedding.shape[1]), dtype=embedding.dtype)
  for i in range(embedding.shape[0]):
    zigma += _embedding[i][:, None].dot(_embedding[i][None, :])
  zigma = zigma / embedding.shape[0]
  u, s, vt = xp.linalg.svd(zigma, full_matrices=True)
  w = u.dot(xp.sqrt(xp.linalg.inv(xp.diag(s))))
  new_embedding = _embedding.dot(w)
  return new_embedding

def whitening_transformation_v2(embedding, seed_index=None):
  xp = get_array_module(embedding)
  if seed_index != None:
    seed_embeddings = embedding[seed_index]
  else:
    seed_embeddings = embedding
  u, s, vt = xp.linalg.svd(seed_embeddings, full_matrices=False)
  w = vt.T.dot(xp.diag(1/s)).dot(vt)
  new_embedding = embedding.dot(w)
  return new_embedding 

def whitening_transformation_v3(embedding):
  xp = get_array_module(embedding)
  def compute_kernel_bias(vecs):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = xp.cov(vecs.T)
    u, s, vh = xp.linalg.svd(cov)
    W = xp.dot(u, xp.diag(1 / xp.sqrt(s)))
    return W, -mu

  def transform_and_normalize(vecs, kernel, bias):
    vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / xp.clip(norms, 1e-8, xp.inf)

  kernel, bias = compute_kernel_bias(embedding)
  new_embedding = transform_and_normalize(embedding, kernel, bias)
  new_embedding = new_embedding.astype(embedding.dtype)
  return new_embedding

def training_noise_reduction(positive_examples):
  new_positive_examples = []
  for s, t in positive_examples:
    if len(set(t) & set("abcdefghijklmnopqrstuvwxytz")) > 0:
      continue
    new_positive_examples.append((s,t))
  return new_positive_examples

def run_dssm_trainning(args):

  print(args)
  
  SL_start_time = time.time()

  # 对于每个src，从tgt单词的cos相似度最高的neg_top_k个单词中随机采样neg_per_pos个
  neg_top_k = args.hard_neg_top_k
  debug = args.debug

  # load up the embeddings
  print("Loading embeddings from disk ...")
  dtype = "float32"
  srcfile = open(args.in_src, encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.in_tar, encoding="utf-8", errors='surrogateescape')
  src_words, x = embeddings.read(srcfile, 30000, dtype=dtype)
  trg_words, z = embeddings.read(trgfile, 30000, dtype=dtype)

  # load the supervised dictionary
  src_word2ind = {word: i for i, word in enumerate(src_words)}
  trg_word2ind = {word: i for i, word in enumerate(trg_words)}
  src_ind2word = {i: word for i, word in enumerate(src_words)}
  trg_ind2word = {i: word for i, word in enumerate(trg_words)}

  # 读入训练集
  pos_examples = []
  f = open(args.train_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          pos_examples.append((src,trg))

  val_examples = []
  f = open(args.val_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          val_examples.append((src,trg))

  train_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in pos_examples] 
  val_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in val_examples]

  print("train data size: ", len(pos_examples))
  print("test data size: ", len(val_examples))

  # pos_examples是word piar的list
  src_indices = [src_word2ind[t[0]] for t in pos_examples]
  trg_indices = [trg_word2ind[t[1]] for t in pos_examples]

  print("unique source words in train data: ", len(set(src_indices)))

  val_src_indices = [src_word2ind[t[0]] for t in val_examples]
  val_trg_indices = [trg_word2ind[t[1]] for t in val_examples] 
  
  # 调用vecmap
  # call artetxe to get the initial alignment on the initial train dict
  xw, zw = x, z

  embeddings.normalize(xw, ['unit', 'center', 'unit'])
  embeddings.normalize(zw, ['unit', 'center', 'unit'])

  with torch.no_grad():
    torch_orig_xw = torch.from_numpy(asnumpy(xw))
    torch_orig_zw = torch.from_numpy(asnumpy(zw))

  if args.use_whitening is not None and args.use_whitening == "pre":
    if args.whitening_data == "train":
      src_indices_for_whitening = sorted(list(set(src_indices))) if args.whitening_sort else src_indices
      tgt_indices_for_whitening = sorted(list(set(trg_indices))) if args.whitening_sort else trg_indices
    else:
      src_indices_for_whitening = None
      tgt_indices_for_whitening = None
    xw = whitening_transformation_v2(xw, src_indices_for_whitening)
    zw = whitening_transformation_v2(zw, tgt_indices_for_whitening)    


  # 生成负例,hard neg examples
  # 返回的是负例word pair的list
  # generate negative examples for the current

  print("Generating negative examples ...")
  generate_negative_func = NEG_SAMPLING_METHOD[args.hard_neg_sampling_method]
  
  src2negtgts, tgt2negsrcs = generate_negative_func(pos_examples, 
                                          src_word2ind, 
                                          trg_word2ind, 
                                          copy.deepcopy(xw), 
                                          copy.deepcopy(zw), 
                                          top_k = neg_top_k, 
                                          num_neg_per_pos = args.hard_neg_per_pos,
                                          hard_neg_random = args.hard_neg_random,
                                          hard_threshold = args.hard_neg_sampling_threshold,
                                          method = args.hard_sim_method)


  if args.use_whitening is not None and args.use_whitening == "post":
    if args.whitening_data == "train":
      src_indices_for_whitening = sorted(list(set(src_indices))) if args.whitening_sort else src_indices
      tgt_indices_for_whitening = sorted(list(set(trg_indices))) if args.whitening_sort else trg_indices
    else:
      src_indices_for_whitening = None
      tgt_indices_for_whitening = None
    xw = whitening_transformation_v2(xw, src_indices_for_whitening)
    zw = whitening_transformation_v2(zw, tgt_indices_for_whitening)  

  if debug:
    # 保存hard neg sample结果用于debug
    debug_sample_wise_neg(src2negtgts, tgt2negsrcs, src_ind2word, trg_ind2word, train_set)
  
  # 去掉score，score是用来debug的
  if not args.hard_neg_random_with_prob:
    for key in src2negtgts:
      src2negtgts[key] = [_[0] for _ in src2negtgts[key]]
    if tgt2negsrcs is not None:
      for key in tgt2negsrcs:
        tgt2negsrcs[key] = [_[0] for _ in tgt2negsrcs[key]]

  print("Training initial classifier ...")

  if debug:
    # 保存最近邻用于debug
    #debug_monolingual_nns(xw, zw, src_ind2word, trg_ind2word)
    pass

  with torch.no_grad():
    torch_xw = torch.from_numpy(asnumpy(xw))
    torch_zw = torch.from_numpy(asnumpy(zw))

  model = DssmTrainer(torch_xw.shape[1], 
                        torch_zw.shape[1], 
                        args.h_dim, 
                        random_neg_per_pos=args.random_neg_per_pos, 
                        epochs=args.train_epochs,
                        eval_every_epoch=args.eval_every_epoch,
                        shuffle_in_train=args.shuffle_in_train,
                        lr=args.lr,
                        train_batch_size=args.train_batch_size,
                        model_save_file=args.model_filename,
                        is_single_tower=args.is_single_tower,
                        hard_neg_per_pos=args.hard_neg_per_pos,
                        hard_neg_random=args.hard_neg_random,
                        loss_metric=args.loss_metric)
                        
  model.fit(torch_xw, torch_zw, train_set, src2negtgts, tgt2negsrcs, val_set, torch_orig_xw, torch_orig_zw)

  print("Writing output to files ...")
  # write res to disk
  # 保存xw, zw
  srcfile = open(args.out_src, mode='w', encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.out_tar, mode='w', encoding="utf-8", errors='surrogateescape')
  embeddings.write(src_words, xw, srcfile)
  embeddings.write(trg_words, zw, trgfile)
  srcfile.close()
  trgfile.close()
  print("SL FINISHED " + str(time.time() - SL_start_time))
  

if __name__ == "__main__":  

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

  parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
  parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
  parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
  parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
  # parameters

  parser.add_argument('--use_whitening', type=str, choices=["pre", "post", None], default=None, help='use whitening transformation before neg sampling as preprocess') 
  parser.add_argument('--whitening_data', type=str, choices=["train", "all"], default="train", help='use whitening transformation before neg sampling as preprocess') 
  parser.add_argument('--whitening_sort', action='store_true', help='sort whitening data')

  # model related para
  parser.add_argument('--is_single_tower', action='store_true', help='use single tower')
  parser.add_argument('--h_dim', type=int, default=300, help='hidden states dim in GNN')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

  parser.add_argument('--hard_neg_per_pos', type=int, default=256, help='number of hard negative examples')
  parser.add_argument('--hard_neg_sampling_method', type=str, choices=NEG_SAMPLING_METHOD.keys(), default='ab', help='method of neg sampling')
  parser.add_argument('--hard_neg_sampling_threshold', type=float, default=-10, help='filter low similarity neg word in sampling hard word')
  parser.add_argument('--hard_neg_random', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--hard_neg_top_k', type=int, default=500, help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_sim_method', type=str, default="cos", help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_neg_random_with_prob', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--random_neg_per_pos', type=int, default=256, help='number of random negative examples')

  
  
  parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
  parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
  parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval epochs')
  parser.add_argument('--shuffle_in_train', action='store_true', help='use shuffle in train')
  parser.add_argument('--loss_metric', type=str, default="cos", help='number of topk examples for select hard neg word')


  
  parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
  parser.add_argument('--debug', action='store_true', help='store debug info')

  args = parser.parse_args()

  run_dssm_trainning(args)

