# This is a modified version by Mladen Karan <m.karan@qmul.ac.uk> of the below code

# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys

import pickle
from classymap import *
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F

BATCH_SIZE = 250

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = numpy
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def calc_monolingual_adj(embedding, threshold=0, method='cos'):
  if method == 'cos':
    adj = embedding.dot(embedding.T)
  elif method == 'csls':
    adj = calc_csls_sim(embedding, embedding, 10, True)
  else:
    adj = None

  _mask = adj > threshold
  print(_mask.sum())
  adj = adj * _mask

  return adj

def load_model(model_file_path):
  with(open(model_file_path, "rb")) as infile:
    model = pickle.load(infile)   
  return model

 
def calc_monolingual_adj(embedding, threshold=0, method='cos'):
  if method == 'cos':
    adj = embedding.dot(embedding.T)
  elif method == 'csls':
    adj = calc_csls_sim(embedding, embedding, 10, True)
  else:
    adj = np.identity(embedding.shape[0])
    return adj

  _mask = adj > threshold
  print(_mask.sum())
  adj = adj * _mask

  return adj

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--model', help = "path to supervised model for reranking", type=str)
    parser.add_argument('--src_lid', help = "Source language id", type=str)
    parser.add_argument('--tar_lid', help = "Target language id", type=str)
    parser.add_argument('--idstring', help = "Idstring used to look up things in the cache.", type=str)
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--eval_result', type=str, default='eval_result.json')

    args = parser.parse_args()
    
    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}
    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    #embeddings.normalize(x, ['unit', 'center', 'unit'])
    #embeddings.normalize(z, ['unit', 'center', 'unit'])

    x_adj = calc_monolingual_adj(x, method='iden')
    z_adj = calc_monolingual_adj(z, method='iden')

    with torch.no_grad():
        torch_xw = torch.from_numpy(asnumpy(x))
        torch_zw = torch.from_numpy(asnumpy(z))
        torch_x_adj = torch.from_numpy(asnumpy(x_adj))
        torch_z_adj = torch.from_numpy(asnumpy(z_adj))

    model = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model.model.to(device)
    model.model.eval()
    torch_xw = torch_xw.to(device)
    torch_zw = torch_zw.to(device)
    torch_x_adj = torch_x_adj.to(device)
    torch_z_adj = torch_z_adj.to(device)


    with torch.no_grad():
        x_h = model.model.src_tower(torch_xw, torch_x_adj)
        z_h = model.model.tgt_tower(torch_zw, torch_z_adj)
        x_h_norm = F.normalize(x_h)
        z_h_norm = F.normalize(z_h)
    new_x = x_h_norm.cpu().numpy()
    new_z = z_h_norm.cpu().numpy()

    x = new_x
    z = new_z



    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = [x.lower().strip() for x in line.split()]
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))



    xp = numpy
    # Find translations
    translation = collections.defaultdict(int)
    scores = collections.defaultdict(int)
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            nn = (-similarities).argsort(axis=1)
            tmp_sim = - similarities
            tmp_sim.sort(axis=1)
            tmp_sim = - tmp_sim
            for k in range(j-i):                
                translation[src[i+k]] = nn[k]
                scores[src[i+k]] = tmp_sim[k]
    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = (-similarities).argsort(axis=1)
            tmp_sim = - similarities
            tmp_sim.sort(axis=1)
            tmp_sim = - tmp_sim
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
                scores[src[i+k]] = tmp_sim[k]
    
    #print(translation[src[0]])
    #print(np.where(translation[src[0]] == 15))
    #print(np.where(translation[src[0]] == 15)[0][0])
    
    result = {}
    src = sorted(src)
    for i in src:
      pred_word = [trg_ind2word[_] for _ in translation[i].tolist()]
      pred_score = scores[i].tolist()
      pred_pair = list(zip(pred_word, pred_score))
      result[src_ind2word[i]] = {
        'gold': [trg_ind2word[_] for _ in src2trg[i]],
        'pred': pred_pair[:100]
      }
    import json
    with open(args.eval_result, 'w', encoding='utf-8') as f:
      json.dump(result, f, ensure_ascii=False, indent=2)
       
    
    # apply supervised if needed

    #positions_old = [np.min([np.where(asnumpy(translation[i]) == x)[0][0]+1 for x in src2trg[i]]) for i in src] 

    positions = [np.min([np.where(asnumpy(translation[i]) == x)[0][0]+1 for x in src2trg[i]]) for i in src] 

    assert len(positions) == len(src)

    p1 = len([p for p in positions if p <= 1]) / len(positions)
    p5 = len([p for p in positions if p <= 5]) / len(positions)
    p10 = len([p for p in positions if p <= 10]) / len(positions)
    mrr = sum([1.0/p for p in positions]) / len(positions)

    print("P1 = %.4f" % (p1))
    print("P5 = %.4f" % (p5))
    print("P10 = %.4f" % (p10))
    print("MRR = %.4f" % (mrr))

    print('Coverage:{0:7.2%}'.format(coverage))

    #print("RAW_OUTPUTS")
    #for p in positions:
      #print(p)
      
 
if __name__ == '__main__':
    main()
