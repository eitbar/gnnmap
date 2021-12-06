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
    def __init__(self, pos_examples, src_w2negs, vocab_size=30000, random_neg_per_pos=1000, hard_neg_per_pos=256, hard_neg_random=True):
      self.lens = len(pos_examples)
      self.vocab_size = vocab_size
      self.datas, self.src2gold = self._build_dataset(pos_examples, src_w2negs)
      self.random_neg_per_pos = random_neg_per_pos
      self.hard_neg_per_pos = hard_neg_per_pos
      self.hard_neg_random = hard_neg_random
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
      if self.hard_neg_random: 
        #print(len(orig))
        hard_neg_sample_list = random.sample(orig[2:], self.hard_neg_per_pos)
        hard_neg_set = set(hard_neg_sample_list)
      else:
        hard_neg_sample_list = orig[2:]
        hard_neg_set = set(hard_neg_sample_list)

      ground_true_set = self.src2gold[orig[0]]
      # 随机采样并与hard、gold去重
      rand_sampling = random.sample(list(range(self.vocab_size)), self.random_neg_per_pos)
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

    def forward(self, node_feat):
        h = node_feat
        for i, v in enumerate(self.mapping_vectors):
            h = self._householderReflection(v, h)
        return h  

class GDSSM(nn.Module):
    
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, is_single_tower=False):
        super(GDSSM, self).__init__()
        self.src_tower = HouseholderTower(src_in_feat_dim, h_feat_dim)
        if is_single_tower == True:
          self.tgt_tower = self._straight_forwoard
        else:
          self.tgt_tower = HouseholderTower(tgt_in_feat_dim, h_feat_dim)

    def _straight_forwoard(self, x):
        return x

    def forward(self, node_feat_src, node_feat_tgt, src_index, tgts_index):
        src_h = self.src_tower(node_feat_src)
        tgt_h = self.tgt_tower(node_feat_tgt)

        src_h_t = src_h[src_index]
        tgt_h_t = tgt_h[tgts_index]

        src_h_t_norm = F.normalize(src_h_t)
        tgt_h_t_norm = F.normalize(tgt_h_t, dim=2)

        sim = torch.matmul(src_h_t_norm.unsqueeze(1), tgt_h_t_norm.transpose(1,2))
        logits = sim.squeeze()     
        return logits


class DssmTrainer:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, 
                  device='gpu', epochs=100, eval_every_epoch=5, lr=0.0001, train_batch_size=256,
                  model_save_file='tmp_model.pickle', is_single_tower=False, 
                  random_neg_per_pos=256, hard_neg_per_pos=256, hard_neg_random=True):
        # train config
        self.epochs = epochs
        self.eval_every_epoch = eval_every_epoch
        self.train_batch_size = train_batch_size
        self.random_neg_per_pos = random_neg_per_pos
        self.model_save_file = model_save_file
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        
        self.hard_neg_per_pos = hard_neg_per_pos
        self.hard_neg_random = hard_neg_random
        # model config
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, is_single_tower)
        #self.loss_func = nn.CrossEntropyLoss()
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
      bpr_loss_batch_mean = bpr_loss.mean()
      return bpr_loss_batch_mean

    def fit(self, src_x, tgt_x, train_set, src_w2negs, val_set):
        
        # for evaluate and debug
        # eval_data_set = train_set
        eval_data_set = val_set   
        eval_src2tgts = collections.defaultdict(set)
        for s, t in eval_data_set:
          eval_src2tgts[s].add(t)
        eval_src = list(set([_[0] for _ in eval_data_set])) 

        model = self.model
        model.to(self.device)
        src_x = src_x.to(self.device)
        tgt_x = tgt_x.to(self.device)

        train_dataset = DssmDatasets(train_set, src_w2negs, 
                                      vocab_size=tgt_x.shape[0], 
                                      random_neg_per_pos=self.random_neg_per_pos,
                                      hard_neg_per_pos=self.hard_neg_per_pos,
                                      hard_neg_random=self.hard_neg_random)

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
                
                logits = model(src_x, tgt_x, src_index, tgts_index)
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
            if e % self.eval_every_epoch == 0 or e == self.epochs - 1:
              model.eval()
              #val_src = train_src
              #val_src2tgts = train_src2tgts
              print(f'In epoch {e} evaluate:')
              acc, scores_result, tgts_result = self.eval(src_x, tgt_x, eval_src, eval_src2tgts)
              
              if best_val_acc < acc:
                if acc[0] - save_best_acc[0] > 0.001:
                  self.save()
                  save_best_acc = acc
                best_val_acc = acc
                best_epoch = e

              print(f"best result at epoch {best_epoch}: {best_val_acc}")
              
    def eval(self, src_x, tgt_x, val_src, val_src2tgts):

      tgts_result = []
      scores_result = []
      eval_bs = 24
      for i in range(0, len(val_src), eval_bs):
        j = min(i + eval_bs, len(val_src))
        bs_pred_result = self.predict(src_x, tgt_x, val_src[i:j])
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

    def predict(self, src_x, tgt_x, test_src, src2tgts_list=None):
        model = self.model
        model.to(self.device)

        src_x = src_x.to(self.device)
        tgt_x = tgt_x.to(self.device)

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
          logits = model(src_x, tgt_x, test_src, test_tgts)
          pred = logits
        return pred

    def save(self):
      print("Saving the best model to disk ...")
      with open("./" + self.model_save_file, "wb") as outfile:
        pickle.dump(self, outfile)