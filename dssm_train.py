import embeddings
from cupy_utils import *
import argparse
import collections
import numpy as np
import sys
import pickle
import time
from random import randrange
from art_wrapper import run_supervised_alignment
from sklearn.utils import shuffle
import torch
from tqdm import tqdm
import random
from graph_utils import calc_csls_sim, topk_mean
from new_dssm import DssmTrainer
import json
import copy

def debug_neg_sampling_record(src_w2negs, src_ind2word, trg_ind2word, src_w2nns, train_set):
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

    exit(0)
    """
    nns_result = []
    for src in src_w2nns:
      src_word = src_ind2word[src]
      nns_trg_word_list = src_w2nns[src]
      nns_trg_word_list = sorted(nns_trg_word_list, key=lambda x:x[1], reverse=True)

      nns_trg_word_list = [trg_ind2word[_[0]] + '(' + "%.5f" % _[1] + ')' for _ in nns_trg_word_list]
      nns_result.append({
        'src_word': src_word,
        'nns_words': ','.join(nns_trg_word_list[:100])
      })
    
    with open('orig_nns_select.json', 'w') as f:
      json.dump(nns_result, f, indent=2, ensure_ascii=False)
    """

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

def debug_graph_structual(xw, zw, src_ind2word, trg_ind2word, src_indices, trg_indices):
    x_adj = calc_monolingual_adj(xw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)
    z_adj = calc_monolingual_adj(zw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)

    train_src_2_edge = {}
    for src_ind in set(src_indices):
        src_adj_index = np.where(x_adj[src_ind] > 0)[0]
        adj_score = x_adj[src_ind][src_adj_index].tolist()
        adj_index = src_adj_index.tolist()
        word_score_pair = list(zip([src_ind2word[_] for _ in adj_index], adj_score))
        train_src_2_edge[src_ind2word[src_ind]] = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in word_score_pair])
    with open('src_word2neiborword.json', 'w') as f:
      json.dump(train_src_2_edge, f, indent=2, ensure_ascii=False)

    train_tgt_2_edge = {}
    for trg_ind in set(trg_indices):
        trg_adj_index = np.where(z_adj[trg_ind] > 0)[0]
        adj_score = z_adj[trg_ind][trg_adj_index].tolist()
        adj_index = trg_adj_index.tolist()
        word_score_pair = list(zip([trg_ind2word[_] for _ in adj_index], adj_score))
        train_tgt_2_edge[trg_ind2word[trg_ind]] = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in word_score_pair])
    with open('tgt_word2neiborword.json', 'w') as f:
      json.dump(train_tgt_2_edge, f, indent=2, ensure_ascii=False)

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


def generate_negative_examples_v1(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos): 
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

  # 为每个pos_src word挑num_neg_per_pos个负例，好像没判重？
  neg_examples = [] 
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
    if num_neg_per_pos != None:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      neg_examples.append((src_ind2w[src_ind], tar_ind2w[neg_wi]))
      src_word2neg_words[src_ind].append((neg_wi, neg_s))

  src_word2nns = collections.defaultdict(list)
  for src_ind in nns:
    for nei_id, nei_score in nns[src_ind]:
      src_word2nns[src_ind].append((nei_id, nei_score))


  return_list = neg_examples
  shuffle(return_list) 
  return return_list, src_word2neg_words, src_word2nns

# method can be "random", "hard" or "mix"
def generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos): 
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
  print(len(tgt_nns))
  # 为每个pos_src word挑num_neg_per_pos个负例
  neg_examples = []  
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
    if num_neg_per_pos != None:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      neg_examples.append((src_ind2w[src_ind], tar_ind2w[neg_wi]))
      src_word2neg_words[src_ind].append((neg_wi, neg_s))

  tgt_word2nns = collections.defaultdict(list)
  for tgt_ind in tgt_nns:
    for nei_id, nei_score in tgt_nns[tgt_ind]:
      tgt_word2nns[tgt_ind].append((nei_id, nei_score))

  return_list = neg_examples
  shuffle(return_list) 
  return return_list, src_word2neg_words, tgt_word2nns

# method can be "random", "hard" or "mix"
def generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos): 
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
  neg_examples = []  
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
    if num_neg_per_pos != None:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    # 加入采样结果
    for neg_wi, neg_s in hard_neg_inds_sample: 
      neg_examples.append((src_ind2w[src_ind], tar_ind2w[neg_wi]))
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
      neg_examples.extend([(src_ind2w[src_nn_ind], tar_ind2w[neg_wi]) for neg_wi in candidate_neg_wi])
      src_word2neg_words[src_nn_ind].extend([(neg_wi, 0) for neg_wi in candidate_neg_wi])

  tgt_word2nns = collections.defaultdict(list)
  for tgt_ind in tgt_nns:
    for nei_id, nei_score in tgt_nns[tgt_ind]:
      tgt_word2nns[tgt_ind].append((nei_id, nei_score))

  return_list = neg_examples
  shuffle(return_list) 
  return return_list, src_word2neg_words, tgt_word2nns

def generate_negative_examples_v4(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos):
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
  _, src_word2neg_words_sim_distri, _ = generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, xsim, zsim, 300, 128)
  _, src_word2neg_words_neg, _ = generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos)
  agg_src_word2_neg_words = {}
  for _src in src_word2neg_words_sim_distri:
    src_word2neg_words_sim_distri[_src] = [(_[0], 0) for _ in src_word2neg_words_sim_distri[_src]]
    src_word2neg_words_neg[_src] = [(_[0], 0) for _ in src_word2neg_words_neg[_src]]
    neg_set = list(set(src_word2neg_words_sim_distri[_src] + src_word2neg_words_neg[_src]))
    agg_src_word2_neg_words[_src] = neg_set
  return _, agg_src_word2_neg_words, _

def generate_negative_examples_v5(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos):
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
  num_neg_per_pos = None
  _, src_word2neg_words_sim_distri, _ = generate_negative_examples_v1(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, xsim, zsim, top_k, num_neg_per_pos)
  del xsim, zsim
  _, src_word2neg_words_neg, _ = generate_negative_examples_v3(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos)
  
  agg_src_word2_neg_words = {}
  mean_len_list = []
  for _src in src_word2neg_words_sim_distri:
    src_word2neg_words_sim_distri[_src] = [(_[0], 0) for _ in src_word2neg_words_sim_distri[_src]]
    src_word2neg_words_neg[_src] = [(_[0], 0) for _ in src_word2neg_words_neg[_src]]
    neg_set = list(set(src_word2neg_words_sim_distri[_src] + src_word2neg_words_neg[_src]))
    agg_src_word2_neg_words[_src] = neg_set
    mean_len_list.append(len(neg_set))
  print(sum(mean_len_list) / len(mean_len_list))
  return _, agg_src_word2_neg_words, _


def calc_monolingual_adj(embedding, threshold=0, method='cos', knn=0):
  xp = get_array_module(embedding)
  if method == 'cos':
    adj = embedding.dot(embedding.T)
  elif method == 'csls':
    adj = calc_csls_sim(embedding, embedding, 10, True)
  else:
    adj = xp.identity(embedding.shape[0])
    return adj

  _mask = adj > threshold
  adj = adj * _mask
  print(_mask.sum())
  if knn > 0:
    sorted_sim_tmp = xp.sort(adj)
    # max_neighbor_number+1 for ignore self
    kth_sim_value_tmp = sorted_sim_tmp[:, -(knn+1)]
    assert kth_sim_value_tmp.shape[0] == adj.shape[0]

    for i in range(adj.shape[0]):
        neighbor_mask = adj[i] >= kth_sim_value_tmp[i]
        adj[i] = adj[i] * neighbor_mask

  print((adj > 0).sum())

  return adj

def calc_csls_translation(x, z, BATCH_SIZE=512, csls_k=10):
  xp = get_array_module(x)
  src = list(range(x.shape[0]))
  translation = collections.defaultdict(int)
  knn_sim_bwd = xp.zeros(z.shape[0])
  for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=csls_k, inplace=True)
  for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
      nn = (-similarities).argsort(axis=1)
      for k in range(j-i):
          translation[src[i+k]] = nn[k]
  return translation

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

def whitening_transformation_v2(embedding):
  xp = get_array_module(embedding)
  u, s, vt = xp.linalg.svd(embedding, full_matrices=False)
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

def whitening_transformation_v4(embedding, seed_index=None):
  if seed_index == None:
    return whitening_transformation_v2(embedding)
  xp = get_array_module(embedding)
  seed_embeddings = embedding[seed_index]
  u, s, vt = xp.linalg.svd(seed_embeddings, full_matrices=False)
  w = vt.T.dot(xp.diag(1/s)).dot(vt)
  new_embedding = embedding.dot(w)
  return new_embedding  

def training_noise_reduction(positive_examples):
  new_positive_examples = []
  for s, t in positive_examples:
    if len(set(t) & set("abcdefghijklmnopqrstuvwxytz")) > 0:
      continue
    new_positive_examples.append((s,t))
  return new_positive_examples

def run_dssm_trainning(args):
  SL_start_time = time.time()


  # 对于每个src，从tgt单词的cos相似度最高的neg_top_k个单词中随机采样neg_per_pos个
  neg_top_k = 500
  neg_per_pos = args.hard_neg_sample
  debug = args.debug
  model_out_filename = args.model_filename
 

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
  # 返回结果xw，zw是embedding矩阵，这个embedding矩阵是映射过后的embedding矩阵
  # call artetxe to get the initial alignment on the initial train dict
  if args.use_origin_emb:
    xw, zw = x, z 
  else:
    print("Starting the Artetxe et al. alignment ...") 
    xw, zw = run_supervised_alignment(src_words, trg_words, x, z, src_indices, trg_indices, supervision = args.art_supervision)

 
  embeddings.normalize(xw, ['unit', 'center', 'unit'])
  embeddings.normalize(zw, ['unit', 'center', 'unit'])

  # 生成负例,hard neg examples
  # 返回的是负例word pair的list
  # generate negative examples for the current 
  neg_per_pos = None
  print("Generating negative examples ...")
  neg_examples, src_w2negs, src_w2nns =  generate_negative_examples_v3(pos_examples, 
                                                                       src_word2ind, 
                                                                       trg_word2ind, 
                                                                       src_ind2word, 
                                                                       trg_ind2word, 
                                                                       copy.deepcopy(xw), 
                                                                       copy.deepcopy(zw), 
                                                                       top_k = neg_top_k, 
                                                                       num_neg_per_pos = neg_per_pos)

  if args.use_whitening:
    print("use_whitening")
    xw = whitening_transformation_v4(xw, sorted(list(set(src_indices))))
    zw = whitening_transformation_v4(zw, sorted(list(set(trg_indices))))

  if debug:
    # 保存hard neg sample结果用于debug
    debug_neg_sampling_record(src_w2negs, src_ind2word, trg_ind2word, src_w2nns, train_set)
  
  for src in src_w2negs:
    src_w2negs[src] = [_[0] for _ in src_w2negs[src]]


  print("Training initial classifier ...")

  if debug:
    # 保存最近邻用于debug
    #debug_monolingual_nns(xw, zw, src_ind2word, trg_ind2word)
    # 保存建图结果用于debug
    # debug_graph_structual(xw, zw, src_ind2word, trg_ind2word, src_indices, trg_indices)
    pass

  with torch.no_grad():
    torch_xw = torch.from_numpy(asnumpy(xw))
    torch_zw = torch.from_numpy(asnumpy(zw))

  if args.graph_method != 'iden':
    x_adj = calc_monolingual_adj(xw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)
    z_adj = calc_monolingual_adj(zw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)
    torch_x_adj = torch.from_numpy(asnumpy(x_adj))
    torch_z_adj = torch.from_numpy(asnumpy(z_adj))
  else:
    torch_x_adj = None
    torch_z_adj = None

  model = DssmTrainer(torch_xw.shape[1], 
                        torch_zw.shape[1], 
                        args.h_dim, 
                        random_neg_sample=args.random_neg_sample, 
                        epochs=args.train_epochs, 
                        lr=args.lr,
                        train_batch_size=args.train_batch_size,
                        model_name=args.model_name,
                        model_save_file=args.model_filename,
                        is_single_tower=args.is_single_tower)

  model.fit(torch_xw, torch_x_adj, torch_zw, torch_z_adj, train_set, src_w2negs, val_set, src_i2w=src_ind2word, tgt_i2w=trg_ind2word, verbose=True)

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
  parser.add_argument('--use_origin_emb', action='store_true', help='use origin fasttext embeddings as model input')
  parser.add_argument('--use_whitening', action='store_true', help='use whitening transformation as preprocess')
  
  

  # graph related para
  parser.add_argument('--graph_method', type=str, default='iden')
  parser.add_argument('--graph_knn', type=int, default=3)
  parser.add_argument('--graph_threshold', type=float, default=0.8)

  # model related para
  parser.add_argument('--model_name', type=str, choices=["gnn", "linear", "hh", "nl_gnn"], default="gnn", help='select model method')
  parser.add_argument('--is_single_tower', action='store_true', help='use single tower')
  parser.add_argument('--h_dim', type=int, default=300, help='hidden states dim in GNN')
  parser.add_argument('--hard_neg_sample', type=int, default=256, help='number of hard negative examples')
  parser.add_argument('--random_neg_sample', type=int, default=256, help='number of random negative examples')
  parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
  parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

  
  parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
  parser.add_argument('--art_supervision', type=str,  default="--supervised", help='Supervision argument to pass on to Artetxe et al. code. Default is "--supervised".')
  parser.add_argument('--debug', action='store_true', help='store debug info')

  args = parser.parse_args()

  run_dssm_trainning(args)

