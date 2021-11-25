import embeddings
from cupy_utils import *
import cupyx
import os
import argparse
import collections
import numpy as np
import sys
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import time
from random import randrange
from art_wrapper import run_supervised_alignment
from sklearn.utils import shuffle
import math
import textdistance
import torch
from tqdm import tqdm
import random
from graph_utils import calc_csls_sim, topk_mean
from new_dssm import Classifier as gnn_Classifier
import json
# src list of indexes from the source vocab to get neighbours for (these index the rows of X)
# X embedding matrix for the source language 
# Z embedding matrix for the target language
# num_NN number of neighbours to generate
# returns a dictionary int --> list(int), each value is a list of length k containing the indexes (from Z) of the nearest neighbours (sorted descending by similarity)
# should be pretty fast on GPU with cuda = "True"
# mode can be "dense" or "sparse"
# supply return_scores = False to make it faster (and you dont need scores)
def get_NN(src, X, Z, num_NN, cuda = False, mode = "dense", batch_size = 100, return_scores = True, fast1NNmode = False):
  # get Z to the GPU once in the beginning (it can be big, seems like a waste to copy it again for every batch)
  if cuda:
    if not supports_cupy():
      print("Error: Install CuPy for CUDA support", file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    if mode == "dense":
      z = xp.asarray(Z)
    elif mode == "sparse":
      z = cupyx.scipy.sparse.csr_matrix(Z)
  else: # not cuda
    z = Z
 
  ret = {}

  # 按batch size计算NN
  for i in range(0, len(src), batch_size):
    print("Starting NN batch " + str(i/batch_size))
    start_time = time.time()

    j = min(i + batch_size, len(src))    
    x_batch_slice = X[src[i:j]]
    
    # get the x part to the GPU if needed
    if cuda:
     if mode == "dense":
        x_batch_slice = xp.asarray(x_batch_slice)
     elif mode == "sparse":
        x_batch_slice = cupyx.scipy.sparse.csr_matrix(x_batch_slice)
    
    similarities = x_batch_slice.dot(z.T)

    if mode == "sparse":
      similarities = similarities.todense()

    # 加符号，这样就是从大到小sort了
    nn = (-similarities).argsort(axis=1)

    for k in range(j-i):
      ind = nn[k,0:num_NN].tolist()
      if mode == "sparse" and not cuda:
        ind = ind[0]

      if return_scores:             
        sim = similarities[k,ind].tolist()
        ret[src[i+k]] = list(zip(ind,sim))
      else:
        ret[src[i+k]] = ind
    print("Time taken " + str(time.time() - start_time))
  return(ret)   

# for each word in src_words computes the num_NN most similar by char ngram cosine, these are the candidate pairs (a total of len(src_words) x num_NN of them)
# returns the candidate pairs sorted in descending order (but not all of them because this is still too much, rather num_output top candidates from the sorted list)
# optionally a resort function can be supplied which will sort (descending) the top num_output candidates one more time (for example if we wanted to sort the top
# examples by edit distance instead of ngram cosine)


# 按照character级别n-gram构造tfidf vector然后计算余弦相似度
def precompute_orthographic_NN(src_words, tar_words, num_NN, num_output = 20000, cuda = False, resort_func = None):
  ret = {}
  tid = TfidfVectorizer(analyzer = "char", min_df = 1, ngram_range = (2,5))
  print("Generating the char n-gram vectors")
  v = tid.fit_transform(src_words + tar_words)
  src_vectors = v[0:len(src_words),:]
  tar_vectors = v[len(src_words):,:]
  
  freq_cutoff = 0.10
  firstN = int(freq_cutoff * len(src_words))
  tar_vectors = tar_vectors[0:firstN,:] # leave only 10% most frequent target words

  src_ind2w = {i: word for i, word in enumerate(src_words)}
  tar_ind2w = {i: word for i, word in enumerate(tar_words)}
 
  print("Sparse matrix multiplication")
  d = get_NN(range(len(src_words)), src_vectors, tar_vectors, num_NN, cuda = cuda, mode = "sparse", batch_size = 1000) # batch of 500 fits on a GPU with 8G RAM
  print("Finished!")

  print("Sorting candidates and finishing up")
  s = []
  for sw in d:
    for tw,c in d[sw]:
        s.append((sw,tw,c))
 
  sorted_s = sorted(s, key = lambda t:t[2], reverse = True) # sort descending by cosine sim
  sorted_s = sorted_s[0:num_output]
  if resort_func is not None:
    sorted_s = sorted(sorted_s, key = lambda t:resort_func(t[0],t[1]), reverse = True)
  ret = [(src_ind2w[k], tar_ind2w[v], c) for k,v,c in sorted_s]
  return(ret, d)

# method can be "random", "hard" or "mix"
# hard examples are wrong pairs that (in spite of being wrong) have high cosine
# mix is half random half hard 

class OrthoNNProvider():
  def __init__(self, path_to_file, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w):
    with(open(path_to_file, "rb")) as infile:
      self.d = pickle.load(infile)
    self.src_ind2w = src_ind2w
    self.tar_ind2w = tar_ind2w
    self.src_w2ind = src_w2ind
    self.tar_w2ind = tar_w2ind
  # ngram做tfidf的topk，按照textdistance的编辑距离取最高的topk
  def get_top_neighbours(self, word, k):
    similar_targets_ind = [t[0] for t in self.d[self.src_w2ind[word]]]
    similar_targets = [self.tar_ind2w[x] for x in similar_targets_ind]
    sim_words = [(w,textdistance.levenshtein.distance(w, word)) for w in similar_targets]
    sorted_sims = sorted(sim_words, key = lambda t:t[1]) # smallest distances first
    return sorted_sims[0:k]
 

# method can be "random", "hard" or "mix"
def generate_negative_examples_v2(positive_examples, src_w2ind, tar_w2ind, src_ind2w, tar_ind2w, x, z, top_k, num_neg_per_pos, num_neg_editdist, ornn): 
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
  nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, mode = "dense", batch_size = 100, return_scores = True, fast1NNmode = False)

  # 为每个pos_src word挑num_neg_per_pos个负例，好像没判重？
  neg_examples = []  
  for src_ind in nns:
    for i in range(num_neg_per_pos):
      sampling_success = False
      while not sampling_success:
        tmp_ind = randrange(top_k)
        rand_neighbour_ind = nns[src_ind][tmp_ind][0]
        rand_score = nns[src_ind][tmp_ind][1]
        if rand_neighbour_ind not in correct_mapping[src_ind] and (rand_neighbour_ind, rand_score) not in src_word2neg_words[src_ind]:
          sampling_success = True
          neg_examples.append((src_ind2w[src_ind], tar_ind2w[rand_neighbour_ind]))
          src_word2neg_words[src_ind].append((rand_neighbour_ind, rand_score))

  # 从编辑距离最近的topk挑选negexample
  neg_examples_ed = []
  if num_neg_editdist > 0:
    for src_w in tqdm(positive_src):
      top_ortho_nns = ornn.get_top_neighbours(src_w, top_k)
      for i in range(num_neg_editdist):
         sampling_success = False
         while not sampling_success:
          rand_neighbour = top_ortho_nns[randrange(top_k)][0]
          if tar_w2ind[rand_neighbour] not in correct_mapping[src_w2ind[src_w]] and tar_w2ind[rand_neighbour] not in src_word2neg_words[src_ind]:
            sampling_success = True
            neg_examples_ed.append((src_w, rand_neighbour))
            src_word2neg_words[src_w2ind[src_w]].append(tar_w2ind[rand_neighbour])

  src_word2nns = collections.defaultdict(list)
  for src_ind in nns:
    for nei_id, nei_score in nns[src_ind]:
      src_word2nns[src_ind].append((nei_id, nei_score))


  return_list = neg_examples + neg_examples_ed
  shuffle(return_list) 
  return return_list, src_word2neg_words, src_word2nns
 
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

def run_dssm_trainning(args):
  SL_start_time = time.time()


  # 对于每个src，从tgt单词的cos相似度最高的neg_top_k个单词中随机采样neg_per_pos个
  neg_top_k = 1000
  neg_per_pos = args.hard_neg_sample
  neg_editdist_per_pos = args.edit_neg_sample
  debug = args.debug
 
  task_name = args.src_lid + "-" + args.tar_lid + "-" + args.idstring
  cng_cachefile = "./cache/ortho_nn_" + task_name + ".pickle"
  cng_nn_file = "./cache/ortho_nn_" + task_name + "-dict.pickle"
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

  # precomputing of orthographic nns, if needed
  # 是否基于编辑距离选取负例
  or_nn_provider = None
  if neg_editdist_per_pos > 0:
    cng_cachefile = "./cache/ortho_nn_" + task_name + ".pickle"
    cng_nn_file = "./cache/ortho_nn_" + task_name + "-dict.pickle"

    if not os.path.isfile(cng_cachefile) or not os.path.isfile(cng_nn_file):
      print("Precomputing ortographic neighbours (will be cached for further calls)")
      neighbours, neighbours_dict = precompute_orthographic_NN(src_words, trg_words, neg_top_k, cuda = True)

      print("Writing top pairs to file ...")
      with open(cng_cachefile, "wb") as outfile: 
        pickle.dump(neighbours, outfile)

      print("Writing precomputed ortho nns to file ...")
      with open(cng_nn_file, "wb") as outfile: 
        pickle.dump(neighbours_dict, outfile)

    #if neg_editdist_per_pos > 0:
    print("Loading the precomputed orthography nns ...")
    or_nn_provider = OrthoNNProvider(cng_nn_file, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word)
 

  # 读入训练集
  src_indices, trg_indices, pos_examples = [], [], []
  f = open(args.train_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          pos_examples.append((src,trg))

  print(len(pos_examples))

  val_examples = []

  f = open(args.val_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          val_examples.append((src,trg))

  print(len(val_examples))

  # pos_examples是word piar的list
  src_indices = [src_word2ind[t[0]] for t in pos_examples]
  trg_indices = [trg_word2ind[t[1]] for t in pos_examples]

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

 
  # 生成负例,hard neg examples
  # 返回的是负例word pair的list
  # generate negative examples for the current 
  print("Generating negative examples ...")
  neg_examples, src_w2negs, src_w2nns =  generate_negative_examples_v2(pos_examples, src_word2ind, trg_word2ind, src_ind2word, trg_ind2word, xw, zw, top_k = neg_top_k, num_neg_per_pos = neg_per_pos, num_neg_editdist = neg_editdist_per_pos, ornn = or_nn_provider)
  
  
  if debug:
    # 保存hard neg sample结果用于debug
    neg_result = []
    for src in src_w2negs:
      src_word = src_ind2word[src]
      neg_trg_word_list = src_w2negs[src]
      neg_trg_word_list = sorted(neg_trg_word_list, key=lambda x:x[1], reverse=True)

      neg_trg_word_list = [trg_ind2word[_[0]] + '(' + "%.5f" % _[1] + ')' for _ in neg_trg_word_list]
      neg_result.append({
        'src_word': src_word,
        'hard_neg_sample_words': ','.join(neg_trg_word_list[:100])
      })
    
    with open('orig_neg_select.json', 'w') as f:
      json.dump(neg_result, f, indent=2, ensure_ascii=False)

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

  for src in src_w2negs:
    src_w2negs[src] = [_[0] for _ in src_w2negs[src]]


  print("Training initial classifier ...")

  embeddings.normalize(xw, ['unit', 'center', 'unit'])
  embeddings.normalize(zw, ['unit', 'center', 'unit'])

  csls_translation = None

  #csls_translation = calc_csls_translation(xw, zw)
  #for _ in csls_translation:
  #  csls_translation[_] = asnumpy(csls_translation[_]).tolist()


  if debug:
    # 保存最近邻 以及 建图结果 用于debug
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

    

  x_adj = calc_monolingual_adj(xw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)
  z_adj = calc_monolingual_adj(zw, threshold=args.graph_threshold, method=args.graph_method, knn=args.graph_knn)


  with torch.no_grad():
    torch_xw = torch.from_numpy(asnumpy(xw))
    torch_zw = torch.from_numpy(asnumpy(zw))
    torch_x_adj = torch.from_numpy(asnumpy(x_adj))
    torch_z_adj = torch.from_numpy(asnumpy(z_adj))


  train_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in pos_examples] 
  val_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in val_examples]
  model = gnn_Classifier(torch_xw.shape[1], 
                        torch_zw.shape[1], 
                        args.h_dim, 
                        train_random_neg_select=args.random_neg_sample, 
                        epochs=args.train_epochs, 
                        lr=args.lr,
                        train_batch_size=args.train_batch_size,
                        model_name=args.model_name)

  model.fit(torch_xw, torch_x_adj, torch_zw, torch_z_adj, train_set, src_w2negs, val_set, csls_translation, src_i2w=src_ind2word, tgt_i2w=trg_ind2word, verbose=True)

  print("Writing output to files ...")
  # write res to disk
  # 保存xw, zw

  srcfile = open(args.out_src, mode='w', encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.out_tar, mode='w', encoding="utf-8", errors='surrogateescape')
  embeddings.write(src_words, xw, srcfile)
  embeddings.write(trg_words, zw, trgfile)
  srcfile.close()
  trgfile.close()
  print("Saving the supervised model to disk ...")
  with open("./" + model_out_filename, "wb") as outfile:
    pickle.dump(model, outfile)
  print(str(args.idstring))
  print("SL FINISHED " + str(time.time() - SL_start_time))
  

if __name__ == "__main__":  

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

  parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
  parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
  parser.add_argument('--src_lid', type=str, help='Source language id.', required = True)
  parser.add_argument('--tar_lid', type=str, help='Target language id.', required = True)
  parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
  parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
  
  
  # graph related para
  parser.add_argument('--graph_method', type=str, default='iden')
  parser.add_argument('--graph_knn', type=int, default=3)
  parser.add_argument('--graph_threshold', type=float, default=0.8)

  # model related para
  parser.add_argument('--model_name', type=str, choices=["gnn", "linear", "hh", "nl_gnn"], default="gnn", help='select model method')
  parser.add_argument('--h_dim', type=int, default=300, help='hidden states dim in GNN')
  parser.add_argument('--hard_neg_sample', type=int, default=256, help='number of hard negative examples')
  parser.add_argument('--edit_neg_sample', type=int, default=0, help='number of hard negative examples based edit distanse')
  parser.add_argument('--random_neg_sample', type=int, default=256, help='number of random negative examples')
  parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
  parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
  parser.add_argument('--use_origin_emb', action='store_true', help='use origin fasttext embeddings as model input')



  
  parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
  parser.add_argument('--idstring', type=str,  default="EXP", help='Special id string that will be included in all generated model and cache files. Default is EXP.')

  parser.add_argument('--art_supervision', type=str,  default="--supervised", help='Supervision argument to pass on to Artetxe et al. code. Default is "--supervised".')

  parser.add_argument('--debug', action='store_true', help='store debug info')

  args = parser.parse_args()

  run_dssm_trainning(args)

