from __future__ import print_function
from PIL import Image
import random, numpy as np
import os, torch, copy
import torch.utils.data as data
from pathlib import Path
from collections import defaultdict

def make_dir(new_dir):
  if not new_dir.exists():
    os.makedirs(new_dir)

class TieredImageNetDataset(data.Dataset):
  def __init__(self, dataset_dir, mode, transform):
    super(TieredImageNetDataset, self).__init__()
    print('Building dataset TieredImageNetLu for [{}] ...'.format(mode))
    self.dataset_dir  = Path(dataset_dir)
    self.data_path    = self.dataset_dir / "{}_wordid_idx_imgs.pth".format(mode)
    self.par_path     = self.dataset_dir / "{}_parents_idx.pth".format(mode)
    self.chi_path     = self.dataset_dir / "{}_children_idx.pth".format(mode)
    self.anc_path     = self.dataset_dir / "{}_ancestors_idx.pth".format(mode)
    self.des_path     = self.dataset_dir / "{}_descendants_idx.pth".format(mode)
    self.lvl_cls_path = self.dataset_dir / "{}_lvl_cls_idx.pth".format(mode)
    self.wordid2idx_path = self.dataset_dir / "{}_wordid2idx.pth".format(mode)
    self.idx2wordid_path = self.dataset_dir / "{}_idx2wordid.pth".format(mode)
    self.transform    = transform  

    if not self.data_path.exists():
      raise RuntimeError('Dataset not found. You can use download=True to download it')
    data = torch.load( self.data_path )
    self.parents       = torch.load(self.par_path)
    self.children      = torch.load(self.chi_path)
    self.ancestors     = torch.load(self.anc_path)
    self.descendants   = torch.load(self.des_path)
    self.lvl_cls       = torch.load(self.lvl_cls_path)
    self.wordid2idx    = torch.load(self.wordid2idx_path)
    self.idx2wordid    = torch.load(self.idx2wordid_path)
    # flat_classes stores all classes wordid in a list
    flat_classes       = [c for lvl, cls in self.lvl_cls.items() for c in cls]
    self.n_classes     = len(list(set(flat_classes)))
    self.few_shot_classes = self.lvl_cls[ max(self.lvl_cls.keys()) ] 
    # these stores the information for every img
    self.all_imgs, self.all_labels, self.all_wordids_idx, self.all_levels = [], [], [], []
    self.cls_idx_dict = defaultdict(list)
    self.wordid_level_label = {}
    index = 0
    for lvl, cls in self.lvl_cls.items():
      for c in cls:
        label = cls.index(c)
        imgs  = data[c]
        self.wordid_level_label[c] = (lvl, label)
        for img in imgs:
          self.all_imgs.append(img)
          self.all_labels.append(label)
          self.all_wordids_idx.append(c)
          self.all_levels.append(lvl)
          self.cls_idx_dict[c].append(index)
          index += 1
    self.cls_idx_dict = dict(self.cls_idx_dict)

    print ('==> Dataset: Found {:} classes and {:} images for all levels'.format(self.n_classes, len(self.all_imgs)))    
    '''
    if mode == "train" and False:
      print("train ancestors are: {}".format(self.ancestors))
      for lvl, cls in self.lvl_cls.items():
        for c in cls:
          dir_name = self.dataset_dir / "original-lvl-{}".format(lvl) / "cls-{}".format(c)
          make_dir(dir_name)
          imgs  = data[c]
          imgs = imgs[:50]
          for idx, img in enumerate(imgs):          
            file_name = dir_name / "lvl{}-cls{}-idx{}.png".format(lvl, c, idx)
            im = Image.fromarray(img[:, :, ::-1])
            im.save(file_name)
      print("save original training imgs done")
    '''

  def __getitem__(self, idx):
    image = self.all_imgs[ idx ].copy()
    pil_image = Image.fromarray( image[:, :, ::-1] )
    if self.transform:
      pil_image = self.transform(pil_image)
    return torch.IntTensor([idx]), pil_image, self.all_labels[idx], self.all_levels[idx], self.all_wordids_idx[idx]


  def __len__(self):
    return len(self.all_imgs)



class FewShotSampler(object):
  
  def __init__(self, labels, levels, few_shot_classes, cls_idx_dict, ancestors, prob_coa, classes_per_it, num_samples, iterations, img_up_bound):
    super(FewShotSampler, self).__init__()
    self.all_labels = labels
    self.all_levels = levels   
    self.cls_idx_dict = cls_idx_dict
    self.ancestors  = ancestors
    self.prob_coa   = prob_coa 
    self.classes_per_it  = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.img_up_bound = img_up_bound
    self.few_shot_classes = few_shot_classes

  def __iter__(self):
    '''
    yield a batch of indexes
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      few_shot_batch, coarse_batch = [], []
      batch_few_shot_classes = random.sample(self.few_shot_classes, cpi) 
      for i, c in enumerate(batch_few_shot_classes):
        img_idxs = self.cls_idx_dict[c]
        few_shot_batch.extend( random.sample(img_idxs, spc))
       
      all_ancestors = []
      for c in batch_few_shot_classes:
        all_ancestors.extend(self.ancestors[c])
      all_ancestors = list(set(all_ancestors)) 

      for anc in all_ancestors:
        anc_all_idx = self.cls_idx_dict[anc]
        #num_samples = min( int(len(anc_all_idx) * self.prob_coa), self.img_up_bound)
        num_samples = self.img_up_bound
        batch_anc_idx = random.sample(anc_all_idx, num_samples)
        coarse_batch.extend(batch_anc_idx) 
      batch = few_shot_batch + coarse_batch
      batch = torch.LongTensor(batch)
      yield batch

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class cls_sampler(object):

  def __init__(self, cls_idx_dict):
    super(cls_sampler, self).__init__()
    self.cls_idx_dict = cls_idx_dict

  def __iter__(self):
    for cls, idx_lst in self.cls_idx_dict.items():
      batch = torch.LongTensor(idx_lst)
      yield batch

  def __len__(self):
    return len(self.cls_idx_dict.keys())

