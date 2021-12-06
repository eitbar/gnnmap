from numpy.core.numeric import False_
from DenseGraphConv import DenseGraphConv
import torch
from torch import embedding, nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
import json
from tqdm import tqdm
import random
import collections
import pickle
#from transformers import WarmupLinearSchedule

class DssmDatasets(Dataset):
    def __init__(self, pos_examples, src_w2negs, vocab_size=30000, random_neg_num=1000, hard_random_flag=True):
      self.lens = len(pos_examples)
      self.vocab_size = vocab_size
      self.datas, self.src2gold = self._build_dataset(pos_examples, src_w2negs)
      self.random_neg_num = random_neg_num
      self.hard_random_flag = hard_random_flag
      random.seed(2021)

    def _build_dataset(self, pos_examples, src_w2negs):
      datas = []
      src2gold = collections.defaultdict(set)
      for pos_src, pos_tgt in pos_examples:
        datas.append([pos_src, pos_tgt] + list(src_w2negs[pos_src]))
        src2gold[pos_src].add(pos_tgt)
      return datas, src2gold

    def __getitem__(self, i):
      # 在getitem的时候随机采样，是为了保证每个epoch采样得到的负例都不相同
      orig = self.datas[i]
      if self.hard_random_flag: 
        #print(len(orig))
        hard_neg_sample_list = random.sample(orig[2:], 256)
        hard_neg_set = set(hard_neg_sample_list)
      else:
        hard_neg_sample_list = orig[2:]
        hard_neg_set = set(hard_neg_sample_list)

      ground_true_set = self.src2gold[orig[0]]
      # 随机采样并与hard、gold去重
      rand_sampling = random.sample(list(range(self.vocab_size)), self.random_neg_num)
      no_dup_random_neg = list(set(rand_sampling) - hard_neg_set - ground_true_set)
      new_item = orig[:2] + hard_neg_sample_list + no_dup_random_neg
      return new_item

    def __len__(self):
      return len(self.datas)

    def collate(self, features):
      # ground truth 总是位于tgts_list的首位
      src_list = [_[0] for _ in features]
      tgts_list = [_[1:] for _ in features]
      labels_list = [0 for _ in features]

      # batch内data cut到同一长度      
      min_neg_size = min([len(_) for _ in tgts_list])
      for i in range(len(tgts_list)):
        tgts_list[i] = tgts_list[i][:min_neg_size]

      # to_tensor
      src_index = torch.tensor(src_list, dtype=torch.long)
      tgts_list_index = torch.tensor(tgts_list, dtype=torch.long)
      labels_list = torch.tensor(labels_list, dtype=torch.long)
      return src_index, tgts_list_index, labels_list

class BaseTower(nn.Module):
    # (AXW)W
    def __init__(self, in_feat_dim, h_feat_dim, dropout=0.2):
        super(BaseTower, self).__init__()
        self.conv = DenseGraphConv(in_feat_dim, h_feat_dim)
        self.proj = nn.Linear(h_feat_dim, h_feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj):
        h = self.conv(adj, node_feat)
        p_h = self.proj(h)
        return p_h

class GraphTower(nn.Module):
    # relu(AXW+B)
    def __init__(self, in_feat_dim, h_feat_dim):
        super(GraphTower, self).__init__()
        self.conv = DenseGraphConv(in_feat_dim, h_feat_dim, bias=True, activation=torch.nn.ReLU())

    def forward(self, node_feat, adj):
        h = self.conv(adj, node_feat)
        return h

class SimpleGraphTower(nn.Module):
    # AXW
    def __init__(self, in_feat_dim, h_feat_dim):
        super(SimpleGraphTower, self).__init__()
        self.conv = DenseGraphConv(in_feat_dim, h_feat_dim)

    def forward(self, node_feat, adj):
        h = self.conv(adj, node_feat)
        return h 

class LinearTower(nn.Module):
    # XW
    def __init__(self, in_feat_dim, h_feat_dim):
        super(LinearTower, self).__init__()
        self.mapping = torch.nn.Linear(in_feat_dim, h_feat_dim, bias=False)

    def forward(self, node_feat, adj):
        h = self.mapping(node_feat)
        return h

class HouseholderTower(nn.Module):
    # XH
    def __init__(self, in_feat_dim, hhr_number):
        super(HouseholderTower, self).__init__()
        print("use HouseholderTower")
        self.mapping_vectors = nn.ParameterList([nn.Parameter(torch.randn((in_feat_dim, 1), requires_grad=True)) 
                                                      for _ in range(hhr_number)])

    def _householderReflection(self, v, x):
        iden = torch.eye(v.shape[0])
        iden = iden.to(x.device)
        qv = iden - torch.matmul(v, v.T) / torch.matmul(v.T, v)
        return torch.matmul(x, qv)

    def forward(self, node_feat, adj):
        h = node_feat
        for i, v in enumerate(self.mapping_vectors):
            h = self._householderReflection(v, h)
        return h  

MODELDICT = {
  "linear" : LinearTower,
  "nl_gnn" : GraphTower,
  "gnn" : SimpleGraphTower,
  "hh" : HouseholderTower
}

class GDSSM(nn.Module):
    
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, model_name="gnn", is_single_tower=False):
        super(GDSSM, self).__init__()
        self.src_tower = MODELDICT[model_name](src_in_feat_dim, h_feat_dim)
        if is_single_tower == True:
          self.tgt_tower = self._straight_forwoard
        else:
          self.tgt_tower = MODELDICT[model_name](src_in_feat_dim, h_feat_dim)

    def _straight_forwoard(self, x, a):
        return x

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


class DssmTrainer:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, 
                  device='gpu', epochs=100, lr=0.0001, train_batch_size=256, random_neg_sample=512, 
                  model_name="gnn", model_save_file='tmp_model.pickle', is_single_tower=False):
        # train config
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.random_neg_sample = random_neg_sample
        self.model_save_file = model_save_file
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        # model config
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, model_name, is_single_tower)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
      #for i, _ in enumerate(diff.detach().cpu().numpy().tolist()):
      #  print(i, _[:20])
      bpr_loss_batch_mean = bpr_loss.mean()
      return bpr_loss_batch_mean

    def fit(self, src_x, src_a, tgt_x, tgt_a, train_set, src_w2negs, val_set, src_i2w=None, tgt_i2w=None, verbose=False):
        
        # for evaluate and debug
        # eval_data_set = train_set
        eval_data_set = val_set   
        eval_src2tgts = collections.defaultdict(set)
        for s, t in eval_data_set:
          eval_src2tgts[s].add(t)
        eval_src = list(set([_[0] for _ in eval_data_set])) 
 
        if src_i2w is None or tgt_i2w is None:
          verbose = False

        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        tgt_x = tgt_x.to(self.device)
        if tgt_a != None and src_a != None:
          src_a = src_a.to(self.device)
          tgt_a = tgt_a.to(self.device)

        train_dataset = DssmDatasets(train_set, src_w2negs, 
                                      vocab_size=tgt_x.shape[0], 
                                      random_neg_num=self.random_neg_sample)

        train_dataloader = DataLoader(train_dataset, 
                                batch_size=self.train_batch_size,
                                shuffle=False, 
                                collate_fn=train_dataset.collate)
        
        optimizer = self.optimizer
        loss_func = self._bpr_loss_func
        #loss_func = self.loss_func
        #scheduler = self.scheduler 

        best_val_acc = [0, 0, 0, 0, 0]
        save_best_acc = [0, 0, 0, 0, 0]
        best_epoch = 0
        global_step = 0
        total_step = ((len(train_set) + self.train_batch_size - 1) // self.train_batch_size) * self.epochs
        

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
                print('In epoch {}, step: {}, loss: {:.5f}, lr: {:.8f} '.format(
                  e, step, loss, optimizer.state_dict()['param_groups'][0]['lr']))

            # evaluate test set
            if e % 1 == 0 or e == self.epochs - 1:
              model.eval()
              #val_src = train_src
              #val_src2tgts = train_src2tgts
              print(f'In epoch {e} evaluate:')
              acc, scores_result, tgts_result = self.eval(src_x, src_a, tgt_x, tgt_a, eval_src, eval_src2tgts)
              
              if best_val_acc < acc:
                if acc[0] - save_best_acc[0] > 0.001:
                  self.save()
                  save_best_acc = acc
                best_val_acc = acc
                best_epoch = e
                # for debug
                if verbose:
                  src2pred_result_top100 = {}
                  for i, s in enumerate(eval_src):
                    si_word = src_i2w[s]
                    pred_scores = scores_result[i][:50]
                    pred_word = [tgt_i2w[_] for _ in tgts_result[i][:50]]
                    src2pred_result_top100[si_word] = {
                      'gold': [tgt_i2w[_] for _ in eval_src2tgts[s]],
                      'pred': list(zip(pred_word, pred_scores))
                      }
                  with open('tmp.json', 'w', encoding='utf-8') as f:
                    json.dump(src2pred_result_top100, f, ensure_ascii=False, indent=2)
              print(f"best result at epoch {best_epoch}: {best_val_acc}")
              
    def eval(self, src_x, src_a, tgt_x, tgt_a, val_src, val_src2tgts):

      tgts_result = []
      scores_result = []
      eval_bs = 24
      for i in range(0, len(val_src), eval_bs):
        j = min(i + eval_bs, len(val_src))
        bs_pred_result = self.predict(src_x, src_a, tgt_x, tgt_a, val_src[i:j])
        #bs_tgts_result = torch.argsort(bs_pred_result, descending=True, dim=1).cpu().numpy().tolist()
        bs_score_result, bs_tgts_result = torch.sort(bs_pred_result, descending=True, dim=1)
        bs_score_result = bs_score_result.cpu().numpy().tolist()
        bs_tgts_result = bs_tgts_result.cpu().numpy().tolist()
        tgts_result = tgts_result + bs_tgts_result
        scores_result = scores_result + bs_score_result

      acc = []
      for k in [1, 5, 10, 50, 100]:
        pred_src2tgts = {}
        for i, s in enumerate(val_src):
          pred_src2tgts[s] = set(tgts_result[i][:k])
        count = 0
        for s in val_src:
          if len(pred_src2tgts[s] & val_src2tgts[s]) > 0:
            count += 1
        print(f'top_{k} acc: {count / len(val_src)}')
        acc.append(count / len(val_src))
      return acc, scores_result, tgts_result

    def predict(self, src_x, src_a, tgt_x, tgt_a, test_src, src2tgts_list=None):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        tgt_x = tgt_x.to(self.device)
        if tgt_a != None and src_a != None:
          src_a = src_a.to(self.device)
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
          pred = logits
        return pred

    def save(self):
      print("Saving the best model to disk ...")
      with open("./" + self.model_save_file, "wb") as outfile:
        pickle.dump(self, outfile)