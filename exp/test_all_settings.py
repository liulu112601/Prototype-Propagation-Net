import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt
from scipy.interpolate import spline
plt.switch_backend('agg')

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import TieredImageNetDataset, FewShotSampler
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from training_strategy import leveltrain, graphtrain_v2, test_no_hierarchy 
import models 


def train_model(lr_scheduler, emb_model, att_par, att_chi, pp_buffer, prop_proto, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support, test_setting):
  acc_lst, ci95_lst = [], []
  ratio_lst = np.arange(0,1,0.1)
  ratio_data = {}
  ratio_dir = Path("ratio-data-record") 
  if not os.path.exists(ratio_dir):
    os.makedirs(ratio_dir)
  for par_ratio in range(0,10): 
    losses, acc1, ci95 = test_no_hierarchy(None, par_ratio, emb_model, att_par, att_chi, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support, pp_buffer, prop_proto)
    acc_lst.append(acc1.avg)
    ci95_lst.append(ci95)
  ratio_data['acc_lst'] = acc_lst
  ratio_data['ci95_lst'] = ci95_lst
  if "simple" in args.dataset_root:
    data = "W-I-Pure"
  elif "addition" in args.dataset_root:
    data = "W-I-Mix"
  else: raise ValueError("invalid dataset")
  if "graph" in args.training_strategy:
    strategy = "graph"
  elif "level" in args.training_strategy:
    strategy = "level"
  else: raise ValueError("invalid training strategy {}".format(args.training_strategy))
  ratio_data['data'] = data
  ratio_data['strategy'] = strategy
  ratio_data['way']  = args.classes_per_it_val
  ratio_data['shot'] = args.num_support_val
  ratio_save_path = ratio_dir / "{}way{}shot_{}_{}_ratio_data.pth".format(args.classes_per_it_val, args.num_support_val, data, strategy)
  torch.save(ratio_data, ratio_save_path)
  max_idx = np.argmax(acc_lst)
  acc = acc_lst[max_idx]; ci95 = ci95_lst[max_idx]
  logger.print("the best acc is {}+-{}, acc on different ratio has been saved to {}".format(acc, ci95, ratio_save_path))
  
  return losses, acc1


def run(args, emb_model, att_par, att_chi, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, hierarchy_info_train, hierarchy_info_test, train_pp_buffer, test_pp_buffer):
  args = deepcopy(args)
  start_time = time.time()
  epoch_time = AverageMeter()
  best_acc, arch  = 0, args.arch 
  train_ancestors, train_parents, train_descendants, train_children = hierarchy_info_train
  test_ancestors, test_parents, test_descendants, test_children = hierarchy_info_test

 
  with torch.no_grad():
    logger.print("-----------init_test_pp for no-hier_data testing----------")
    test_pp_buffer.init_test_buffer_no_hier_data(train_pp_buffer)
    all_level_classes = list( range(test_pp_buffer.pp_running.shape[0]) )   
    prop_proto = models.get_att_proto(all_level_classes, train_parents, test_pp_buffer.pp_running, att_par, args.n_hop)
    test_loss, test_acc1 = train_model(None, emb_model, att_par, att_chi, test_pp_buffer, prop_proto, criterion, None, logger, test_dataloader, hierarchy_info_test, test_dataloader.dataset.wordid_level_label, -1, args, 'test', args.num_support_val, "with-few-shot-cls")
    logger.print ('*[TEST-Best]* ==> Test-Loss: {:.4f} Test-Acc1: {:.3f}, the TEST-ACC in record is {:.4f}'.format(test_loss.avg, test_acc1.avg, best_acc))

    logger.print("-----------init_test_pp for no-hier_data no few-shot classes testing----------")
    test_pp_buffer.init_test_buffer_no_hier_data_v2(train_pp_buffer)
    all_level_classes = list( range(test_pp_buffer.pp_running.shape[0]) )   
    prop_proto = models.get_att_proto(all_level_classes, train_parents, test_pp_buffer.pp_running, att_par, args.n_hop)
    test_loss, test_acc1 = train_model(None, emb_model, att_par, att_chi, test_pp_buffer, prop_proto, criterion, None, logger, test_dataloader, hierarchy_info_test, test_dataloader.dataset.wordid_level_label, -1, args, 'test', args.num_support_val, "no-few-shot-cls")
    logger.print ('*[TEST-Best]* ==> Test-Loss: {:.4f} Test-Acc1: {:.3f}, the TEST-ACC in record is {:.4f}'.format(test_loss.avg, test_acc1.avg, best_acc))

  return None 
  

def main():
  args = get_parser()
  # create logger
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
  logger = Logger(args.log_dir, args.manual_seed)
  logger.print ("args :\n{:}".format(args))

  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  torch.backends.cudnn.benchmark = True
  np.random.seed(args.manual_seed)
  torch.manual_seed(args.manual_seed)
  torch.cuda.manual_seed(args.manual_seed)

  # create dataloader
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_transform   = transforms.Compose([transforms.Resize(150), transforms.RandomResizedCrop(112), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
  train_dataset    = TieredImageNetDataset(args.dataset_root, 'train', train_transform)
  train_sampler    = FewShotSampler(train_dataset.all_labels, train_dataset.all_levels, train_dataset.few_shot_classes, train_dataset.cls_idx_dict, train_dataset.ancestors, args.prob_coa, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, args.iterations, args.img_up_bound)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.workers)

  test_transform   = transforms.Compose([transforms.Resize(150), transforms.CenterCrop(112), transforms.ToTensor(), normalize])
  test_dataset     = TieredImageNetDataset(args.dataset_root, 'test', test_transform)
  test_sampler     = FewShotSampler(test_dataset.all_labels, test_dataset.all_levels, test_dataset.few_shot_classes, test_dataset.cls_idx_dict, test_dataset.ancestors, args.prob_coa, args.classes_per_it_val, args.num_support_val + args.num_query_val, 600, args.img_up_bound)
  test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.workers)
  # children include both training and testing classes children info
  hierarchy_info_train   = [train_dataset.ancestors, train_dataset.parents, train_dataset.descendants, train_dataset.children]  
  hierarchy_info_test    = [test_dataset.ancestors, test_dataset.parents, test_dataset.descendants, test_dataset.children]  
  # create model
  emb_model  = models.__dict__[args.arch]()
  emb_model  = torch.nn.DataParallel(emb_model).cuda()
  if args.arch == "resnet18":
    fea_dim = 512
  elif args.arch == "simpleCnn":
    fea_dim = 3136 
  att_par  = models.__dict__[args.att_arch](fea_dim)
  att_par  = torch.nn.DataParallel(att_par).cuda()
  att_chi  = models.__dict__[args.att_arch](fea_dim)
  att_chi  = torch.nn.DataParallel(att_chi).cuda()
  train_pp_buffer = models.pp_buffer(train_dataset, fea_dim).cuda() 
  test_pp_buffer  = models.pp_buffer(test_dataset, fea_dim).cuda()
  logger.print ("emb_model:::\n{:}".format(emb_model))
  logger.print ("att_par:::\n{:}".format(att_par))
  logger.print ("att_chi:::\n{:}".format(att_chi))
  logger.print ("train_pp_buffer:::\n{:}".format(train_pp_buffer))
  logger.print ("test_pp_buffer:::\n{:}".format(test_pp_buffer))
  criterion = nn.CrossEntropyLoss().cuda()

  info_path = '{logdir:}/info-last-{arch:}.pth'.format(logdir=str(logger.baseline_classifier_dir), arch=args.arch)
  params = [p for p in emb_model.parameters()] + [p for p in att_par.parameters()] + [p for p in att_chi.parameters()]
  optimizer    = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=args.lr_gamma, step_size=args.lr_step)

  [l,r] = str(logger.baseline_classifier_dir).split("TEST-") 
  model_best_path = '{:}/model_{:}_best.pth'.format(l+r, args.arch)
  best_checkpoint = torch.load(model_best_path)
  emb_model.load_state_dict( best_checkpoint['emb_state_dict'] )
  att_par.load_state_dict( best_checkpoint['att_par_state_dict'] )
  att_chi.load_state_dict( best_checkpoint['att_chi_state_dict'] )
  train_pp_buffer.load_state_dict( best_checkpoint['train_pp_buffer'] )  
  test_pp_buffer.load_state_dict( best_checkpoint['test_pp_buffer'] )  
  logger.print("restore model from {}".format(model_best_path))
  # use original model to test
  info_path = run(args, emb_model, att_par, att_chi, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, hierarchy_info_train, hierarchy_info_test, train_pp_buffer, test_pp_buffer)
  logger.print ('save into {:}'.format(info_path))

if __name__ == '__main__':
  main()
