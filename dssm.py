from dgl.nn import DenseGraphConv
import torch
from torch import embedding, nn
import dgl
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import argparse
import math
from random import randrange
from sklearn.utils import shuffle
import collections
from transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader, Sampler, Dataset, SequentialSampler


def generate_negative_examples(positive_examples, src_w2ind, tar_w2ind, x, z, method = "random"): 
  l = len(positive_examples)
  if method == "mix":
    num_rand, num_hard = math.floor(l / 2), math.ceil(l / 2) # floor/ceil so the sum is still == l
  elif method == "random":
    num_rand, num_hard = l, 0
  elif method == "hard":
    num_rand, num_hard = 0, l
  else:
    raise Exception("Unsupported method for negative sampling.")

  rand_list, hard_list = [], []
  positive_src = [t[0] for t in positive_examples]
  positive_tar = [t[1] for t in positive_examples]

  # generate the random examples (num_rand of them)
  for i in range(num_rand):
    success = False
    while(not success):
      src_ind, tar_ind = randrange(l), randrange(l)
      if src_ind != tar_ind: # when the indexes are the same that is a positive example
        rand_list.append((positive_src[src_ind], positive_tar[tar_ind]))
        success = True
   
  # generate the hard examples (num_hard of them, or skip it if we dont need  them) 
    
  if num_hard > 0:
    pos_src_word_indexes = [src_w2ind[i] for i in positive_src]
    pos_tar_word_indexes = [tar_w2ind[i] for i in positive_tar]
    pos_src_embeddings = x[pos_src_word_indexes,:]
    pos_tar_embeddings = z[pos_tar_word_indexes,:]
    print("Starting dot prod")
    similarities = pos_src_embeddings.dot(pos_tar_embeddings.T)
    print("Finished dot prod")
    l = len(positive_src)

    sflat = similarities.flatten()
    sind = (-sflat).argsort()
     
    current = 0
    for i in range(num_hard): # stupid but works fast enough, do n_hard argmaxes on the array (argmax is very fast and there will only ever be a few thousand of them needed)
      success = False
      while not success:
        ind = sind[current].item()
        current += 1
        si, ti = int(ind / l), ind % l
        if si != ti:
          hard_list.append((positive_src[si], positive_tar[ti]))
          success = True
  
  ret_list = rand_list + hard_list
  shuffle(ret_list)  
  return(ret_list)

class dssmdatasets(Dataset):
    def __init__(self, pos_examples, src_w2negs):
      self.lens = len(pos_examples) 
      self.datas = self._build_dataset(pos_examples, src_w2negs)

    def _build_dataset(self, pos_examples, src_w2negs):
      datas = []
      for pos_src, pos_tgt in pos_examples:
        datas.append([pos_src, pos_tgt, src_w2negs[pos_src]])
      return datas

    def __getitem__(self, i):
      return self.datas[i]

    def __len__(self, i):
      return len(self.datas)

    def collate(self, features):
      pos_src_list = [_[0] for _ in features]
      tgts_list = []
      labels_list = []
      for f in features:
        tgts_list.append([f[1]] + f[2])
        labels_list.append([1] + [0] * len(f[2]))

      min_neg_size = min([len(_) for _ in tgts_list])
      for tgts in tgts_list:
        tgts = tgts[:min_neg_size]

      src_index = torch.tensor(pos_src_list, dtype=torch.long)
      tgts_list_index = torch.tensor(tgts_list, dtype=torch.long)
      labels_list = torch.tensor(labels_list, dtype=torch.long)
      return src_index, tgts_list_index, labels_list



      

class BaseTower(nn.Module):

    def __init__(self, in_feat_dim, h_feat_dim, dropout=0.2):
        super(BaseTower, self).__init__()
        self.conv = DenseGraphConv(in_feat_dim, h_feat_dim)
        self.proj = nn.Linear(h_feat_dim, h_feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj):
        h = self.conv(adj, node_feat)
        h = self.dropout(h)
        p_h = self.proj(h)
        return p_h


class GDSSM(nn.Module):
    
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim):
        super(GDSSM, self).__init__()
        self.src_tower = BaseTower(src_in_feat_dim, h_feat_dim)
        self.tgt_tower = BaseTower(tgt_in_feat_dim, h_feat_dim)

    def forward(self, node_feat_src, adj_src, node_feat_tgt, adj_tgt, node_pair, node_pair_val=None):
        src_h = self.src_tower(node_feat_src, adj_src)
        tgt_h = self.src_tower(node_feat_tgt, adj_tgt)

        src_h_t = src_h[node_pair[0]]
        tgt_h_t = tgt_h[node_pair[1]]

        src_h_t_norm = F.normalize(src_h_t)
        tgt_h_t_norm = F.normalize(tgt_h_t)

        cosine = torch.cosine_similarity(src_h_t_norm, tgt_h_t_norm, dim=1, eps=1e-8)
        pred = F.sigmoid(cosine)
        pred_v = None

        if node_pair_val is not None:
            src_h_v = src_h[node_pair_val[0]]
            tgt_h_v = tgt_h[node_pair_val[1]]

            src_h_v_norm = F.normalize(src_h_v)
            tgt_h_v_norm = F.normalize(tgt_h_v)

            cosine_v = torch.cosine_similarity(src_h_v_norm, tgt_h_v_norm, dim=1, eps=1e-8)
            pred_v = F.sigmoid(cosine_v)      

        return pred, pred_v


class Classifier:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, device='gpu', epochs=5000):
        print(device == 'gpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim)
        self.epochs = epochs
        self.loss_func = F.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        #self.scheduler = WarmupLinearSchedule(
        #self.optimizer, warmup_steps=0, t_total=epochs)

    def fit(self, src_x, src_a, tgt_x, tgt_a, data):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        src_a = src_a.to(self.device)
        tgt_x = tgt_x.to(self.device)
        tgt_a = tgt_a.to(self.device)

        for _ in data:
            data[_] = data[_].to(self.device)

        train_x = data['train_x']
        train_y = data['train_y']

        val_x = data['val_x'] if 'val_x' in data else None
        val_y = data['val_y'] if 'val_y' in data else None

        print(src_x.device, src_a.device, tgt_x.device, tgt_a.device, train_x.device, val_x.device)


        
        optimizer = self.optimizer
        loss_func = self.loss_func
        #scheduler = self.scheduler 

        best_val_acc = 0

        for e in range(self.epochs):
            # Forward
            pred, pred_val = model(src_x, src_a, tgt_x, tgt_a, train_x, val_x)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = loss_func(pred, train_y)

            # Compute accuracy on training/validation/test
            train_acc = ((pred > 0.5) == train_y).float().mean()
            if val_x is not None and pred_val is not None:
                val_acc = ((pred_val > 0.5) == val_y).float().mean()
                if best_val_acc < val_acc:
                    best_val_acc = val_acc

            # Backward
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), train acc: {:.3f} '.format(
                    e, loss, val_acc, best_val_acc, train_acc))

    def predict(self, src_x, src_a, tgt_x, tgt_a, data):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        src_a = src_a.to(self.device)
        tgt_x = tgt_x.to(self.device)
        tgt_a = tgt_a.to(self.device)

        for _ in data:
            data[_] = data[_].to(self.device)

        test_x = data['test_x']  

        pred, _ = model(src_x, src_a, tgt_x, tgt_a, test_x)
        return pred





            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

    parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
    parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)

    parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
    parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
    


    #parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
    #parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
    #parser.add_argument('--src_lid', type=str, help='Source language id.', required = True)
    #parser.add_argument('--tar_lid', type=str, help='Target language id.', required = True)
    #parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
    #parser.add_argument('--idstring', type=str,  default="EXP", help='Special id string that will be included in all generated model and cache files. Default is EXP.')

    parser.add_argument('--scoring', type=str,  default='f1_macro', help='Scoring type for the classifier, can be any string valid in sklearn. Default is f1_macro.')
    parser.add_argument('--num_iterations', type=int,  default=10, help='Number of self learning iterations to run. Default is 10.')
    parser.add_argument('--examples_to_pool', type=int,  default=5000, help='Number of examples to pool in each self learning iteration. Default is 5000.')
    parser.add_argument('--examples_to_add', type=int,  default=500, help='Number of examples from the pool to add to the train set in each self learning iteration. Default is 500.')
    parser.add_argument('--use_mnns_pooler', type=int,  default=0, help='Whether to MNN stochastic pooler instead of regular MNN pooler (1 for yes 0 for no). Default is 0.')
    parser.add_argument('--use_classifier', type=int,  default=1, help='Whether to use the classifier to rerank pooled candidates. Default is 1.')

    parser.add_argument('--use_edit_dist', type=int,  default=1, help='Whether to use edit distance features (1 for yes 0 for no). Default is 1.')
    parser.add_argument('--use_aligned_cosine', type=int,  default=1, help='Whether to use cosing distance in aligned space features (1 for yes 0 for no). Default is 1.')
    parser.add_argument('--use_ngrams', type=int,  default=1, help='Whether to use ngram overlap features (1 for yes 0 for no). Default is 1.')
    parser.add_argument('--use_full_bert', type=int,  default=0, help='Whether to use bert based features (1 for yes 0 for no). Default is 0.')
    parser.add_argument('--use_pretrained_bpe', type=int,  default=0, help='Whether to use pretrained BPE features (1 for yes 0 for no). Default is 0.')
    parser.add_argument('--use_frequencies', type=int,  default=1, help='Whether to use frequency features (1 for yes 0 for no). Default is 1.')
    parser.add_argument('--use_aligned_pca', type=int,  default=1, help='Whether to use PCA reduced embeddings in the aligned space as features (1 for yes 0 for no). Default is 1.')
    parser.add_argument('--use_char_ngrams', type=int,  default=0, help='Whether to use character ngrams as features (1 for yes 0 for no). Default is 0.')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')

    parser.add_argument('--art_supervision', type=str,  default="--supervised", help='Supervision argument to pass on to Artetxe et al. code. Default is "--supervised".')
    parser.add_argument('--checkpoint_steps', type=int,  default=-1, help='A checkpoint will be saved every checkpoint_steps iterations. -1 to skip saving checkpoints. Default is -1.')


    args = parser.parse_args()

    import embeddings
    # 读入训练集
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

    src_indices, trg_indices, pos_examples = [], [], []
    f = open(args.train_dict, encoding=args.encoding, errors='surrogateescape')
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

    print(len(src2trg))
    print(sum([len(src2trg[_]) for _ in src2trg]))
    print(coverage)
    
    f = open(args.val_dict, encoding=args.encoding, errors='surrogateescape')
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

    print(len(src2trg))
    print(sum([len(src2trg[_]) for _ in src2trg]))
    print(coverage)
    """
    # pos_examples是word piar的list
    src_indices = [src_word2ind[t[0]] for t in pos_examples]
    trg_indices = [trg_word2ind[t[1]] for t in pos_examples]

    val_src_indices = [src_word2ind[t[0]] for t in val_examples]
    val_trg_indices = [trg_word2ind[t[1]] for t in val_examples]    

    # generate negative examples for the current 
    print("Generating negative examples ...")
    neg_examples = generate_negative_examples(pos_examples, src_word2ind, trg_word2ind, x, z, method = "mix")
    
    train_set = pos_examples + neg_examples
    train_set = [[src_word2ind[_s] for _s, _t in train_set], [trg_word2ind[_t] for _s, _t in train_set] ]
    train_labs = [1]*len(pos_examples) + [0]*len(neg_examples)

    val_set = val_examples
    val_set = [[src_word2ind[_s] for _s, _t in val_set], [trg_word2ind[_t] for _s, _t in val_set] ]
    val_labs = [1]*len(val_examples)

    train_src = sorted([_[0] for _ in train_set])
    train_tgt = sorted([_[1] for _ in train_set])
    val_src = sorted([_[0] for _ in val_set])
    val_tgt = sorted([_[1] for _ in val_set])  

    data = {
        'train_x': torch.tensor(train_set, dtype=torch.long),
        'train_y': torch.tensor(train_labs, dtype=torch.float32),
        'val_x': torch.tensor(val_set, dtype=torch.long),
        'val_y': torch.tensor(val_labs, dtype=torch.float32)
    }
    print(torch.cuda.is_available())

    x = torch.from_numpy(x)
    z = torch.from_numpy(z)
    x_sim = torch.from_numpy(x_sim)
    z_sim = torch.from_numpy(z_sim)

    c = Classifier(300, 300, 300)
    c.fit(x, x_sim, z, z_sim, data)
    """



    








    

    