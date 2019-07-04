import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from configs import AverageMeter
from datasets import cls_sampler
from copy import deepcopy

class pp_buffer(nn.Module):
  def __init__(self, dataset, fea_dim):
    super(pp_buffer, self).__init__()
    sampler = cls_sampler(dataset.cls_idx_dict)
    self.dataloader = torch.utils.data.DataLoader( dataset, batch_sampler=sampler, num_workers=16 ) 
    self.lvl_cls = dataset.lvl_cls
    n_class    = dataset.n_classes
    pp_running = torch.zeros((n_class, fea_dim)) 
    self.n_class = n_class
    self.fea_dim = fea_dim
    #count = torch.zeros((n_class, 1))
    #sum_pp = torch.zeros((n_class, fea_dim))
    
    self.register_buffer("pp_running", pp_running)
    #self.register_buffer("count", count)
    #self.register_buffer("sum_pp", sum_pp)

  def reset_buffer(self, emb_model):
    with torch.no_grad():
      for batch_idx, (img_idx, imgs, labels, levels, wordids) in enumerate(self.dataloader):
        cls_pp_lst = []
        for start in range(0, len(imgs), 200):
          end = min(len(imgs), start+200)
          pp = emb_model(imgs[start: end])
          cls_pp_lst.append(pp)
        cls_pp = torch.mean( torch.cat(cls_pp_lst, dim=0), dim=0 )
        self.pp_running[wordids[0]] = cls_pp
      torch.cuda.empty_cache()

  # only for test_buffer, init values in test_buffer using train_buffer 
  def init_test_buffer(self, emb_model, train_pp_buffer, train_wordid2idx, test_idx2wordid):
    self.pp_running = torch.zeros((self.n_class, self.fea_dim)).cuda()
    with torch.no_grad():
      for batch_idx, (img_idx, imgs, labels, levels, wordids) in enumerate(self.dataloader):
        if set(levels) == {7}:
          continue
        else:
          cls_pp_lst = []
          for start in range(0, len(imgs), 200):
            end = min(len(imgs), start+200)
            pp = emb_model(imgs[start: end])
            cls_pp_lst.append(pp)
          cls_pp = torch.mean( torch.cat(cls_pp_lst, dim=0), dim=0 )
          wordid = test_idx2wordid[wordids[0].item()]
          if wordid in train_wordid2idx.keys():
            cls_pp = 0.5 * cls_pp + 0.5 * train_pp_buffer.pp_running[ train_wordid2idx[wordid] ]
            self.pp_running[wordids[0]] = cls_pp
      torch.cuda.empty_cache()

  '''
  def init_test_buffer_no_data(self, train_pp_buffer, train_wordid2idx, test_idx2wordid):
    self.pp_running = torch.zeros((self.n_class, self.fea_dim)).cuda()
    with torch.no_grad():
      train_wordids = train_wordid2idx.keys()
      for idx, wordid in test_idx2wordid.items():
        if wordid in train_wordids: 
          train_idx = train_wordid2idx[wordid]
          cls_pp = train_pp_buffer.pp_running[train_idx]
          self.pp_running[idx] = cls_pp 
  '''

  def init_test_buffer_no_hier_data(self, train_pp_buffer):
    # all training classes include fine classes 
    with torch.no_grad():
      self.pp_running = deepcopy( train_pp_buffer.pp_running )

  def init_test_buffer_no_hier_data_v2(self, train_pp_buffer):
    train_lvl_cls_idx = train_pp_buffer.lvl_cls
    # all training classes except fine classes
    with torch.no_grad():
      ws_cls = [ train_lvl_cls_idx[idx] for idx in range(0, max(train_lvl_cls_idx.keys())) ]
      ws_cls = [item for sublist in ws_cls for item in sublist] 
      self.pp_running = deepcopy( train_pp_buffer.pp_running[ws_cls] ) 
