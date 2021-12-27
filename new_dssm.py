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


def get_rand_list_with_p(a, size, p):
  p = 1/(1+np.exp(-np.array(p)))
  p = p / np.sum(p)
  sample_list = np.random.choice(a, size, False, p)
  return sample_list.tolist()

class DssmDatasets(Dataset):
    def __init__(self, pos_examples, src2negtgts, tgt2negsrcs=None, vocab_size=30000, 
                random_neg_per_pos=1000, hard_neg_per_pos=256, hard_neg_random=True):
      self.lens = len(pos_examples)
      self.vocab_size = vocab_size
      self.datas, self.src2gold, self.tgt2gold, \
        self.sample2negtgts, self.sample2negsrcs = self._build_dataset(
                                                  pos_examples, src2negtgts, tgt2negsrcs)
      self.random_neg_per_pos = random_neg_per_pos
      self.hard_neg_per_pos = hard_neg_per_pos
      self.hard_neg_random = hard_neg_random
      random.seed(2021)

    def _build_dataset(self, pos_examples, src2negtgts, tgt2negsrcs):
      datas = []
      src2gold = collections.defaultdict(set)
      tgt2gold = collections.defaultdict(set)

      sample2negtgts = collections.defaultdict(set)
      sample2negsrcs = collections.defaultdict(set)
      if tgt2negsrcs is None:
        sample2negsrcs = None
      
      # 如果src在sample2negtgts说明是word-wise的
      # 将word-wise调整为sample-wise，为了之后的处理更加统一
      is_word_wise = pos_examples[0][0] in src2negtgts 
      if not is_word_wise:
        sample2negtgts = src2negtgts
        sample2negsrcs = tgt2negsrcs
      else:
        for pos_src, pos_tgt in pos_examples:
          sample2negtgts[(pos_src, pos_tgt)] = src2negtgts[pos_src]
          if tgt2negsrcs is not None:
            sample2negsrcs[(pos_src, pos_tgt)] = tgt2negsrcs[pos_tgt]

      for pos_src, pos_tgt in pos_examples:
        datas.append([pos_src, pos_tgt])
        src2gold[pos_src].add(pos_tgt)
        tgt2gold[pos_tgt].add(pos_src)

      return datas, src2gold, tgt2gold, sample2negtgts, sample2negsrcs

    def __getitem__(self, i):
      # 在getitem的时候随机采样，是为了保证每个epoch采样得到的负例都不相同
      orig = self.datas[i]
      src = orig[0]
      tgt = orig[1]
      negtgts = list(self.sample2negtgts[(src, tgt)])
      negtgts_prob = None
      if len(negtgts) > 0:
        # 有给定的概率，按照给定概率采样
        if not isinstance(negtgts[0], int):
          negtgts_prob = [_[1] for _ in negtgts]
          negtgts = [_[0] for _ in negtgts]

        if self.hard_neg_random:
          if negtgts_prob is not None:
            hard_neg_tgts_list = get_rand_list_with_p(negtgts, min(self.hard_neg_per_pos, len(negtgts)), negtgts_prob)
          else:
            #hard_neg_tgts_list = random.sample(negtgts, min(self.hard_neg_per_pos, len(negtgts)))
            hard_neg_tgts_list = negtgts[:self.hard_neg_per_pos]
          hard_neg_tgts_set = set(hard_neg_tgts_list)
        else:
          hard_neg_tgts_list = negtgts
          hard_neg_tgts_set = set(hard_neg_tgts_list)
      else:
        hard_neg_tgts_list = []
        hard_neg_tgts_set = set()

      rand_sampling_tgts = random.sample(list(range(self.vocab_size)), self.random_neg_per_pos)
      no_dup_random_tgts = list(set(rand_sampling_tgts) - hard_neg_tgts_set - self.src2gold[src])
      combi_tgts = hard_neg_tgts_list + no_dup_random_tgts

      # 双向
      combi_srcs = []
      if self.sample2negsrcs is not None:
        negsrcs = list(self.sample2negsrcs[(src, tgt)])
        if len(negsrcs) > 0:
          negsrcs_prob = None
          if not isinstance(negsrcs[0], int):
            negsrcs_prob = [_[1] for _ in negsrcs]
            negsrcs = [_[0] for _ in negsrcs]        
          if self.hard_neg_random: 
            #print(len(orig[2:]))
            if negsrcs_prob is not None:
              hard_neg_srcs_list = get_rand_list_with_p(negsrcs, min(self.hard_neg_per_pos, len(negsrcs)), negsrcs_prob)
            else:
              hard_neg_srcs_list = random.sample(negsrcs, min(self.hard_neg_per_pos, len(negsrcs)))
            hard_neg_srcs_set = set(hard_neg_srcs_list)
          else:
            hard_neg_srcs_list = negsrcs
            hard_neg_srcs_set = set(hard_neg_srcs_list)
        else:
          hard_neg_srcs_list = []
          hard_neg_srcs_set = set()

        rand_sampling_srcs = random.sample(list(range(self.vocab_size)), self.random_neg_per_pos)
        no_dup_random_srcs = list(set(rand_sampling_srcs) - hard_neg_srcs_set - self.tgt2gold[tgt])
        combi_srcs = hard_neg_srcs_list + no_dup_random_srcs
      # 每个item是一个tuple，由两个list组成，第一个是src的list，第二个是tgt的list
      # 正例永远位于list首位
      new_item = ([src] + combi_srcs, [tgt] + combi_tgts)
      return new_item

    def __len__(self):
      return len(self.datas)

    def collate(self, features):
      # ground truth 总是位于tgts_list的首位
      srcs_list = [_[0] for _ in features]
      tgts_list = [_[1] for _ in features]
      labels_list = [[0, 0] for _ in features]

      # batch内data cut到同一长度      
      min_neg_tgt_size = min([len(_) for _ in tgts_list])
      for i in range(len(tgts_list)):
        tgts_list[i] = tgts_list[i][:min_neg_tgt_size]

      # batch内data cut到同一长度      
      min_neg_src_size = min([len(_) for _ in srcs_list])
      for i in range(len(srcs_list)):
        srcs_list[i] = srcs_list[i][:min_neg_src_size]      

      # to_tensor
      srcs_list_index = torch.tensor(srcs_list, dtype=torch.long)
      tgts_list_index = torch.tensor(tgts_list, dtype=torch.long)
      labels_list = torch.tensor(labels_list, dtype=torch.long)
      return srcs_list_index, tgts_list_index, labels_list

    def update_hard_neg(self, similarity_x2y, similarity_y2x=None):
      neg_candi_size = max([len(_) + 3 for _ in self.sample2negtgts.values()])
      src2hard_neg_candi = {}
      _, index_x2y = torch.sort(similarity_x2y, descending=True, dim=1)
      index_x2y = index_x2y.cpu().numpy().tolist()
      for src in self.src2gold:
        neg_candi = index_x2y[src][:neg_candi_size]
        neg_candi = [_ for _ in neg_candi if _ not in self.src2gold[src]]
        src2hard_neg_candi[src] = neg_candi
      for (pos_src, pos_tgt) in self.datas:
        self.sample2negtgts[(pos_src, pos_tgt)] = src2hard_neg_candi[pos_src]
      if self.sample2negsrcs is not None and similarity_y2x is not None:
        _, index_y2x = torch.sort(similarity_y2x, descending=True, dim=1)
        index_y2x = index_y2x.cpu().numpy().tolist()
        tgt2hard_neg_candi = {}
        for tgt in self.tgt2gold:
          neg_candi = index_y2x[tgt][:neg_candi_size]
          neg_candi = [_ for _ in neg_candi if _ not in self.tgt2gold[tgt]]
          tgt2hard_neg_candi[tgt] = neg_candi
        for (pos_src, pos_tgt) in self.datas:
          self.sample2negsrcs[(pos_src, pos_tgt)] = tgt2hard_neg_candi[pos_tgt]
              
    def update_hard_neg_v2(self, similarity_x2y_cos, similarity_y2x_cos=None, similarity_x2y_csls=None, similarity_y2x_csls=None):
      neg_candi_size = max([len(_) + 3 for _ in self.sample2negtgts.values()])
      src2hard_neg_candi = {}
      _, index_x2y = torch.sort(similarity_x2y_cos, descending=True, dim=1)
      index_x2y = index_x2y.cpu().numpy().tolist()
      for src in self.src2gold:
        neg_candi = index_x2y[src][:neg_candi_size]
        neg_candi = [_ for _ in neg_candi if _ not in self.src2gold[src]]
        src2hard_neg_candi[src] = neg_candi

      _, index_x2y = torch.sort(similarity_x2y_csls, descending=True, dim=1)
      index_x2y = index_x2y.cpu().numpy().tolist()
      for src in self.src2gold:
        neg_candi = index_x2y[src][:neg_candi_size]
        neg_candi = [_ for _ in neg_candi if _ not in self.src2gold[src]]

        neg_candi1 = list(set(src2hard_neg_candi[src][:int(neg_candi_size // 2)] + neg_candi[:int(neg_candi_size // 2)]))
        neg_candi2 = list(set(src2hard_neg_candi[src][int(neg_candi_size // 2):] + neg_candi[int(neg_candi_size // 2):]))
        for _ in neg_candi2:
          if len(neg_candi1) >= neg_candi_size:
            break
          if _ not in neg_candi1:
            neg_candi1.append(_)
        src2hard_neg_candi[src] = neg_candi1   

      for (pos_src, pos_tgt) in self.datas:
        self.sample2negtgts[(pos_src, pos_tgt)] = src2hard_neg_candi[pos_src]
      if self.sample2negsrcs is not None and similarity_y2x_cos is not None:
        _, index_y2x = torch.sort(similarity_y2x_cos, descending=True, dim=1)
        index_y2x = index_y2x.cpu().numpy().tolist()
        tgt2hard_neg_candi = {}
        for tgt in self.tgt2gold:
          neg_candi = index_y2x[tgt][:neg_candi_size]
          neg_candi = [_ for _ in neg_candi if _ not in self.tgt2gold[tgt]]
          tgt2hard_neg_candi[tgt] = neg_candi

        _, index_y2x = torch.sort(similarity_y2x_csls, descending=True, dim=1)
        index_y2x = index_y2x.cpu().numpy().tolist()
        for tgt in self.tgt2gold:
          neg_candi = index_y2x[tgt][:neg_candi_size]
          neg_candi = [_ for _ in neg_candi if _ not in self.tgt2gold[tgt]]
          neg_candi1 = list(set(tgt2hard_neg_candi[src][:int(neg_candi_size // 2)] + neg_candi[:int(neg_candi_size // 2)]))
          neg_candi2 = list(set(tgt2hard_neg_candi[src][int(neg_candi_size // 2):] + neg_candi[int(neg_candi_size // 2):]))
          for _ in neg_candi2:
            if len(neg_candi1) >= neg_candi_size:
              break
            if _ not in neg_candi1:
              neg_candi1.append(_)
          tgt2hard_neg_candi[src] = neg_candi1        


        for (pos_src, pos_tgt) in self.datas:
          self.sample2negsrcs[(pos_src, pos_tgt)] = tgt2hard_neg_candi[pos_tgt]
      




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

    def _get_hidden(self, node_feat, t="src"):
        if t == "src":
          return self.src_tower(node_feat)
        else:
          return self.tgt_tower(node_feat)

    def forward(self, node_feat_src, node_feat_tgt, srcs_index, tgts_index):
        src_h = self.src_tower(node_feat_src)
        tgt_h = self.tgt_tower(node_feat_tgt)

        pos_src = src_h[srcs_index[:, 0]]
        pos_tgt = tgt_h[tgts_index[:, 0]]

        src_list = src_h[srcs_index]
        tgt_list = tgt_h[tgts_index]

        pos_src_norm = F.normalize(pos_src)
        tgt_list_norm = F.normalize(tgt_list, dim=2)
        sim_src2tgt = torch.matmul(pos_src_norm.unsqueeze(1), tgt_list_norm.transpose(1,2))

        pos_tgt_norm = F.normalize(pos_tgt)
        src_list_norm = F.normalize(src_list, dim=2)
        sim_tgt2src = torch.matmul(pos_tgt_norm.unsqueeze(1), src_list_norm.transpose(1,2))

        src_hidden_norm = F.normalize(src_h)
        tgt_hidden_norm = F.normalize(tgt_h)
        # size n1 * n2
        sim_matrix = torch.matmul(src_hidden_norm, tgt_hidden_norm.transpose(0,1))
        sim_src_topk, _ = torch.topk(sim_matrix, 10, dim=1)
        rt = torch.mean(sim_src_topk, dim=1)
        sim_tgt_topk, _ = torch.topk(sim_matrix.transpose(0,1), 10, dim=1)
        rs = torch.mean(sim_tgt_topk, dim=1)

        srcs_rt = rt[srcs_index]
        tgts_rs = rs[tgts_index]

        logits_src2tgt = sim_src2tgt.squeeze() * 2 - srcs_rt[:, 0:1] - tgts_rs
        logits_tgt2src = sim_tgt2src.squeeze() * 2 - tgts_rs[:, 0:1] - srcs_rt    
        return logits_src2tgt, logits_tgt2src


class DssmTrainer:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, 
                  device='gpu', epochs=100, eval_every_epoch=5, lr=0.0001, train_batch_size=256,
                  model_save_file='tmp_model.pickle', is_single_tower=False, shuffle_in_train=True,
                  random_neg_per_pos=256, hard_neg_per_pos=256, hard_neg_random=True, 
                  update_neg_every_epoch=1, random_warmup_epoches=0, loss_metric="cos"):
        # train config
        self.epochs = epochs
        self.eval_every_epoch = eval_every_epoch
        self.train_batch_size = train_batch_size
        self.random_neg_per_pos = random_neg_per_pos
        self.model_save_file = model_save_file
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        self.loss_metric = loss_metric

        self.hard_neg_per_pos = hard_neg_per_pos
        self.hard_neg_random = hard_neg_random
        self.shuffle_in_train = shuffle_in_train
        self.update_neg_every_epoch = update_neg_every_epoch
        self.random_warmup_epoches = random_warmup_epoches
        # model config
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, is_single_tower)
        #self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = WarmupLinearSchedule(
        #self.optimizer, warmup_steps=0, t_total=epochs
        #)

    def _liner_adjust_lr(self, optimizer, total_step, init_lr, end_lr, now_step):
        lr = init_lr - (init_lr - end_lr) * ((total_step - now_step) / total_step)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr

    def _bpr_loss_func(self, logits, labels_index, rt, rs):
        
        new_logits = logits
        pos_si = new_logits[:, 0]
        neg_si = new_logits[:, 1:]
        diff = pos_si[:, None] - neg_si
        bpr_loss = - diff.sigmoid().log().mean(1)
        bpr_loss_batch_mean = bpr_loss.mean()
        return bpr_loss_batch_mean

    def _calc_r_in_csls(self, src_x, tgt_x, knn=10):
        src_hidden = self.model._get_hidden(src_x, "src")
        tgt_hidden = self.model._get_hidden(tgt_x, "tgt")
        src_hidden = src_hidden.detach()
        tgt_hidden = tgt_hidden.detach()

        src_hidden = F.normalize(src_hidden)
        tgt_hidden = F.normalize(tgt_hidden)
        # size n1 * n2
        sim_matrix = torch.matmul(src_hidden, tgt_hidden.transpose(0,1))
        sim_src_topk, _ = torch.topk(sim_matrix, knn, dim=1)
        rt = torch.mean(sim_src_topk, dim=1)
        sim_tgt_topk, _ = torch.topk(sim_matrix.transpose(0,1), knn, dim=1)
        rs = torch.mean(sim_tgt_topk, dim=1)
        return rt, rs, sim_matrix

    def fit(self, src_x, tgt_x, train_set, src2negtgts, tgt2negsrcs, val_set, torch_orig_xw=None, torch_orig_zw=None):
        
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
        if torch_orig_xw is not None:
          torch_orig_xw = torch_orig_xw.to(self.device)
        if torch_orig_zw is not None:
          torch_orig_zw = torch_orig_zw.to(self.device)          

        train_dataset = DssmDatasets(train_set, src2negtgts, tgt2negsrcs,
                                      vocab_size=tgt_x.shape[0], 
                                      random_neg_per_pos=self.random_neg_per_pos,
                                      hard_neg_per_pos=self.hard_neg_per_pos,
                                      hard_neg_random=self.hard_neg_random)

        train_dataloader = DataLoader(train_dataset, 
                                batch_size=self.train_batch_size,
                                shuffle=self.shuffle_in_train, 
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
        
        rs = torch.zeros(tgt_x.shape[0])
        rt = torch.zeros(src_x.shape[0])
        rs = rs.to(self.device)
        rt = rt.to(self.device)
        for e in range(self.epochs):
          # Forward
          if torch_orig_xw is not None and torch_orig_zw is not None:
            rt, rs, sim = self._calc_r_in_csls(torch_orig_xw, torch_orig_zw)
          else:
            rt, rs, sim = self._calc_r_in_csls(src_x, tgt_x)

          if self.update_neg_every_epoch > 0 and e % self.update_neg_every_epoch == 0:
            #if self.loss_metric == "cos":
            sim_x2y_cos = sim
            sim_y2x_cos = sim.transpose(0, 1)
            sim_x2y_csls = sim * 2 - rt[:, None] - rs
            sim_y2x_csls = sim.transpose(0, 1) * 2 - rs[:, None] - rt
            train_dataset.update_hard_neg_v2(sim_x2y_cos, sim_y2x_cos, sim_x2y_csls, sim_y2x_csls)
          
          if e < self.random_warmup_epoches:
            train_dataset.hard_neg_per_pos = 0
            train_dataset.random_neg_per_pos = 512
          else:
            train_dataset.hard_neg_per_pos = 256
            train_dataset.random_neg_per_pos = 256            

          model.train()
          for step, batch in enumerate(train_dataloader):
            srcs_index, tgts_index, labels_index = batch
            srcs_index = srcs_index.to(self.device)
            tgts_index = tgts_index.to(self.device)
            labels_index = labels_index.to(self.device)
            
            logits_src2tgt, logits_tgt2src = model(src_x, tgt_x, srcs_index, tgts_index)

            if step == 0:
              print(srcs_index.cpu()[:, 0].numpy().tolist()[:5])
              print(tgts_index.cpu()[:, :5].numpy().tolist()[:5])
              print(logits_src2tgt.detach().cpu()[:, :5].numpy().tolist()[:5])

            loss1 = loss_func(logits_src2tgt, labels_index, rt[srcs_index[:, 0]], rs[tgts_index])
            loss2 = 0
            if srcs_index.shape[1] > 1:
              loss2 = loss_func(logits_tgt2src, labels_index, rs[tgts_index[:, 0]], rt[srcs_index])
              loss = (loss1 + loss2) / 2
            else:
              loss = loss1
            
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            #self._liner_adjust_lr(optimizer, total_step, 0.01, 0.0001, global_step)
            print('In epoch {}, step: {}, loss: {:.5f}, loss1: {:.5f}, loss2: {:.5f}, lr: {:.8f} '.format(
              e, step, loss, loss1, loss2, optimizer.state_dict()['param_groups'][0]['lr']))
            
            # evaluate test set
          if e % self.eval_every_epoch == 0 or e == self.epochs - 1:
            model.eval()
            #val_src = train_src
            #val_src2tgts = train_src2tgts
            print(f'In epoch {e} evaluate:')
            if torch_orig_xw is not None and torch_orig_zw is not None:
              acc, scores_result, tgts_result = self.eval(torch_orig_xw, torch_orig_zw, eval_src, eval_src2tgts)
            else:
              acc, scores_result, tgts_result = self.eval(src_x, tgt_x, eval_src, eval_src2tgts)

            if best_val_acc < acc:
              if acc[0] - save_best_acc[0] > 0.001:
                self.save()
                save_best_acc = acc
              best_val_acc = acc
              best_epoch = e
            print(f"best result at epoch {best_epoch}: {best_val_acc}")
        self.best_val_acc = best_val_acc
              
    def eval(self, src_x, tgt_x, val_src, val_src2tgts):
      tgts_result = []
      scores_result = []
      eval_bs = 24
      #rt, rs, _ = self._calc_r_in_csls(src_x, tgt_x, knn=10)
      for i in range(0, len(val_src), eval_bs):
        j = min(i + eval_bs, len(val_src))
        bs_pred_result = self.predict(src_x, tgt_x, val_src[i:j])
        #bs_pred_result = bs_pred_result * 2 - rt[val_src[i:j], None] - rs
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
        
        test_src = [[_] for _ in test_src]
        # pad
        max_tgts_len = max([len(_) for _ in test_tgts])
        for i in range(len(test_tgts)):
          test_tgts[i] = test_tgts[i] + [0] * (max_tgts_len - len(test_tgts[i]))

        test_src = torch.tensor(test_src, dtype=torch.long, device=self.device)
        test_tgts = torch.tensor(test_tgts, dtype=torch.long, device=self.device)
        with torch.no_grad():
          logits, _ = model(src_x, tgt_x, test_src, test_tgts)
          pred = logits
        return pred

    def save(self):
      print("Saving the best model to disk ...")
      if self.model_save_file is None:
        print("Save failed for model_save_file para is None !!!!")
      else:
        with open("./" + self.model_save_file, "wb") as outfile:
          pickle.dump(self, outfile)