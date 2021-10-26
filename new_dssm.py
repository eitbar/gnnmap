from DenseGraphConv import DenseGraphConv
import torch
from torch import embedding, nn
import dgl
import numpy as np
from torch._C import device
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, dataloader
import os
import argparse
import math
from tqdm import tqdm
import random
from random import randrange
from sklearn.utils import shuffle
import collections
from transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader, Sampler, Dataset, SequentialSampler

class DssmDatasets(Dataset):
    def __init__(self, pos_examples, src_w2negs, vocab_size=30000, random_neg_num=1000):
      self.lens = len(pos_examples)
      self.vocab_size = vocab_size 
      self.datas = self._build_dataset(pos_examples, src_w2negs)
      self.random_neg_num = random_neg_num
      random.seed(2021)

    def _build_dataset(self, pos_examples, src_w2negs):
      datas = []
      for pos_src, pos_tgt in pos_examples:
        datas.append([pos_src, pos_tgt, src_w2negs[pos_src]])
      return datas

    def __getitem__(self, i):
      return self.datas[i]

    def __len__(self):
      return len(self.datas)

    def collate(self, features):
      pos_src_list = [_[0] for _ in features]
      tgts_list = []
      labels_list = []
      for f in features:
        random_neg_list = []
        while(len(random_neg_list) < self.random_neg_num):
          random_neg_wi = random.randint(0, self.vocab_size - 1)
          if random_neg_wi in f[2] or random_neg_wi == f[1]:
            continue
          else:
            random_neg_list.append(random_neg_wi)

        tgts_list.append([f[1]] + f[2] + random_neg_list)
        labels_list.append(0)

      min_neg_size = min([len(_) for _ in tgts_list])
      for i in range(len(tgts_list)):
        tgts_list[i] = tgts_list[i][:min_neg_size]

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

    def forward(self, node_feat_src, adj_src, node_feat_tgt, adj_tgt, src_index, tgts_index):
        src_h = self.src_tower(node_feat_src, adj_src)
        tgt_h = self.tgt_tower(node_feat_tgt, adj_tgt)

        src_h_t = src_h[src_index]
        tgt_h_t = tgt_h[tgts_index]

        src_h_t_norm = F.normalize(src_h_t)
        tgt_h_t_norm = F.normalize(tgt_h_t, dim=2)

        sim = torch.matmul(src_h_t_norm.unsqueeze(1), tgt_h_t_norm.transpose(1,2))
        logits = sim.squeeze()     
        return logits


class Classifier:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, device='gpu', epochs=20, lr=0.005, train_batch_size=256, train_random_neg_select=512):
        print(device == 'gpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim)
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_random_neg_select = train_random_neg_select
        #self.scheduler = WarmupLinearSchedule(
        #self.optimizer, warmup_steps=0, t_total=epochs)

    def _liner_adjust_lr(self, optimizer, total_step, init_lr, end_lr, now_step):
      lr = init_lr - (init_lr - end_lr) * ((total_step - now_step) / total_step)
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def _bpr_loss_func(self, logits, labels_index):
      pos_si = logits[:, 0]
      neg_si = logits[:, 1:]
      diff = pos_si[:, None] - neg_si
      bpr_loss = - diff.sigmoid().log().mean(1)
      bpr_loss_batch_mean = bpr_loss.mean()
      return bpr_loss_batch_mean

    def fit(self, src_x, src_a, tgt_x, tgt_a, pos_examples, src_w2negs, val_examples):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        src_a = src_a.to(self.device)
        tgt_x = tgt_x.to(self.device)
        tgt_a = tgt_a.to(self.device)

        val_src2tgts = collections.defaultdict(set)
        for s, t in val_examples:
          val_src2tgts[s].add(t)
        val_src = list(set([_[0] for _ in val_examples]))

        train_src2tgts = collections.defaultdict(set)
        for s, t in pos_examples:
          train_src2tgts[s].add(t)
        train_src = list(set([_[0] for _ in pos_examples]))        


        train_dataset = DssmDatasets(pos_examples, src_w2negs, random_neg_num=self.train_random_neg_select)
        train_dataloader = DataLoader(train_dataset, 
                                batch_size=self.train_batch_size,
                                shuffle=False, 
                                collate_fn=train_dataset.collate)
        
        optimizer = self.optimizer
        loss_func = self._bpr_loss_func
        #loss_func = self.loss_func
        #scheduler = self.scheduler 

        best_val_acc = 0
        global_step = 0
        total_step = ((len(pos_examples) + self.train_batch_size - 1) // self.train_batch_size) * self.epochs
        

        for e in range(self.epochs):
            # Forward
            model.train()
            for step, batch in enumerate(train_dataloader):
                src_index, tgts_index, labels_index = batch
                src_index = src_index.to(self.device)
                tgts_index = tgts_index.to(self.device)
                labels_index = labels_index.to(self.device)

                logits = model(src_x, src_a, tgt_x, tgt_a, src_index, tgts_index)
                loss = loss_func(logits, labels_index)

                loss.backward()
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                #self._liner_adjust_lr(optimizer, total_step, 0.01, 0.0001, global_step)

                print('In epoch {}, step: {}, loss: {:.5f}, lr: {:.8f} '.format(e, step, loss, optimizer.state_dict()['param_groups'][0]['lr']))

            # evaluate test set
            if e % 10 == 0 or e == self.epochs - 1:
              model.eval()
              #val_src = train_src
              #val_src2tgts = train_src2tgts
              print(f'In epoch {e} evaluate:')
              eval_bs = 24
              tgts_result = []
              for i in range(0, len(val_src), eval_bs):
                j = min(i + eval_bs, len(val_src))
                bs_pred_result = self.predict(src_x, src_a, tgt_x, tgt_a, val_src[i:j])
                bs_tgts_result = torch.argsort(bs_pred_result, descending=True, dim=1).cpu().numpy().tolist()
                tgts_result = tgts_result + bs_tgts_result

              for k in [1, 5, 10]:
                pred_src2tgts = {}
                for i, s in enumerate(val_src):
                  pred_src2tgts[s] = set(tgts_result[i][:k])
                count = 0
                for s in val_src:
                  if len(pred_src2tgts[s] & val_src2tgts[s]) > 0:
                    count += 1
                print(f'top_{k} acc: {count / len(val_src)}')
              

    def predict(self, src_x, src_a, tgt_x, tgt_a, test_src, src2tgts_list=None):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        src_a = src_a.to(self.device)
        tgt_x = tgt_x.to(self.device)
        tgt_a = tgt_a.to(self.device)

        if src2tgts_list is None:
          test_tgts = [list(range(tgt_x.shape[0]))] * len(test_src)
        else:
          test_tgts = [src2tgts_list[_] for _ in test_src]
        
        # pad
        max_tgts_len = max([len(_) for _ in test_tgts])
        for i in range(len(test_tgts)):
          test_tgts[i] = test_tgts[i] + [0] * (max_tgts_len - len(test_tgts[i]))

        test_src = torch.tensor(test_src, dtype=torch.long, device=self.device)
        test_tgts = torch.tensor(test_tgts, dtype=torch.long, device=self.device)
        with torch.no_grad():
          logits = model(src_x, src_a, tgt_x, tgt_a, test_src, test_tgts)
          #pred = logits.squeeze()
          pred = logits
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



    








    

    