import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import TieredImageNetDataset, FewShotSampler
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from training_strategy import leveltrain 
import models 


def train_model(lr_scheduler, emb_model, att_par, att_chi, pp_buffer, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support):
  training_strategy = deepcopy(args.training_strategy)
  if "level" in training_strategy:
    print("level wise train")
    losses, acc1, acc_base, acc_par, acc_chi = leveltrain(lr_scheduler, emb_model, att_par, att_chi, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support, pp_buffer)
  else: raise(ValueError("undefined strategy : {}".format(training_strategy)))

  return losses, acc1, acc_base, acc_par, acc_chi


def run(args, emb_model, att_par, att_chi, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, hierarchy_info_train, hierarchy_info_test, train_pp_buffer, test_pp_buffer):
  args = deepcopy(args)
  start_time = time.time()
  epoch_time = AverageMeter()
  best_acc, arch  = 0, args.arch 
  model_best_path = '{:}/model_{:}_best.pth'.format(str(logger.baseline_classifier_dir), arch)
  model_lst_path  = '{:}/model_{:}_lst.pth'.format(str(logger.baseline_classifier_dir), arch)

  train_ancestors, train_parents, train_descendants, train_children = hierarchy_info_train
  test_ancestors, test_parents, test_descendants, test_children = hierarchy_info_test

  if os.path.isfile(model_lst_path):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    emb_model.load_state_dict(checkpoint['emb_state_dict'])
    att_par.load_state_dict(checkpoint['att_par_state_dict'])
    att_chi.load_state_dict(checkpoint['att_chi_state_dict'])
    train_pp_buffer.load_state_dict(checkpoint['train_pp_buffer'])
    test_pp_buffer.load_state_dict(checkpoint['test_pp_buffer'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch = 0

  start_decay_epoch = args.start_decay_epoch
  for iepoch in range(start_epoch, args.epochs):

    time_str = convert_secs2time(epoch_time.val * (args.epochs- iepoch), True)
    logger.print ('Train {:04d} / {:04d} Epoch, [LR={:6.4f} ~ {:6.4f}], {:}'.format(iepoch, args.epochs, min(lr_scheduler.get_lr()), max(lr_scheduler.get_lr()), time_str))
    
    if iepoch >= start_decay_epoch:
      lr_sch = lr_scheduler
    else:
      lr_sch = None
        
    train_loss, acc1, acc_base, acc_par, acc_chi = train_model(lr_sch, emb_model, att_par, att_chi, train_pp_buffer, criterion, optimizer, logger, train_dataloader, hierarchy_info_train, train_dataloader.dataset.wordid_level_label, iepoch, args, 'train', args.num_support_tr)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

    info = {'epoch'           : iepoch,
            'args'            : deepcopy(args),
            'finish'          : iepoch+1==args.epochs,
            'emb_state_dict'  : emb_model.state_dict(),
            'att_par_state_dict': att_par.state_dict(),
            'att_chi_state_dict': att_chi.state_dict(),
            'train_pp_buffer' : train_pp_buffer.state_dict(),
            'test_pp_buffer'  : test_pp_buffer.state_dict(),
            'optimizer'       : optimizer.state_dict(),
            'scheduler'       : lr_scheduler.state_dict(),
            }
    try:
      torch.save(info, model_lst_path)
      logger.print(' -->> joint-arch :: save into {:}\n'.format(model_lst_path))
    except PermmisionError:
      print("unsucceful write log") 

    if iepoch % args.reset_interval == 0:
      logger.print("reset training buffers, the running mean prototypes are updated according to the concurrent CNN weight ....")
      train_pp_buffer.reset_buffer(emb_model)

    with torch.no_grad():
      if iepoch % args.test_interval == 0 :
        logger.print ('---------init_test_pp-------------')
        # write from train_pp to test_pp
        test_pp_buffer.init_test_buffer(emb_model, train_pp_buffer, train_dataloader.dataset.wordid2idx, test_dataloader.dataset.idx2wordid)
        test_loss, test_acc1, test_acc_base, test_acc_par, test_acc_chi  = train_model(None, emb_model, att_par, att_chi, test_pp_buffer, criterion, None, logger, test_dataloader, hierarchy_info_test, test_dataloader.dataset.wordid_level_label, -1, args, 'test', args.num_support_val)
        logger.print ('Epoch: {:04d} / {:04d} || Train-Loss: {:.4f} Train-Acc: {:.3f} || Test-Loss: {:.4f} Test-Acc1: {:.3f}, Test-Acc_base: {:.3f}, Test-Acc_par: {:.3f}, Test-Acc_chi: {:.3f}\n'.format(iepoch, args.epochs, train_loss.avg, acc1.avg, test_loss.avg, test_acc1.avg, test_acc_base.avg, test_acc_par.avg, test_acc_chi.avg))
        if test_acc1.avg >= best_acc:
          try:
            torch.save(info, model_best_path)
          except PermissionError: pass
          best_acc = test_acc1.avg

  best_checkpoint = torch.load(model_best_path)
  emb_model.load_state_dict( best_checkpoint['emb_state_dict'] )
  att_par.load_state_dict( best_checkpoint['att_par_state_dict'] )
  att_chi.load_state_dict( best_checkpoint['att_chi_state_dict'] )
  train_pp_buffer.load_state_dict( best_checkpoint['train_pp_buffer'] )  
  test_pp_buffer.load_state_dict( best_checkpoint['test_pp_buffer'] )  
 
  with torch.no_grad():
    logger.print("-----------init_test_pp----------")
    test_pp_buffer.init_test_buffer(emb_model, train_pp_buffer, train_dataloader.dataset.wordid2idx, test_dataloader.dataset.idx2wordid)
    test_loss, test_acc1, test_acc_base, test_acc_par, test_acc_chi = train_model(None, emb_model, att_par, att_chi, test_pp_buffer, criterion, None, logger, test_dataloader, hierarchy_info_test, test_dataloader.dataset.wordid_level_label, -1, args, 'test', args.num_support_val)
  logger.print ('*[TEST-Best]* ==> Test-Loss: {:.4f} Test-Acc1: {:.3f}, Test-Acc_base: {:.3f}, Test-Acc_par: {:.3f}, Test-Acc_chi: {:.3f}, the TEST-ACC in record is {:.4f}'.format(test_loss.avg, test_acc1.avg, test_acc_base.avg, test_acc_par.avg, test_acc_chi.avg, best_acc))

  info_path = '{logdir:}/info-last-{arch:}.pth'.format(logdir=str(logger.baseline_classifier_dir), arch=args.arch)
  torch.save({'emb_state_dict': best_checkpoint['emb_state_dict'], 'att_par_state_dict': best_checkpoint['att_par_state_dict'], 'att_chi_state_dict': best_checkpoint['att_chi_state_dict'], 'train_pp_buffer': best_checkpoint['train_pp_buffer'], 'test_pp_buffer': best_checkpoint['test_pp_buffer']}, info_path)
  return info_path
  

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

  info_path = run(args, emb_model, att_par, att_chi, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader, hierarchy_info_train, hierarchy_info_test, train_pp_buffer, test_pp_buffer)
  logger.print ('save into {:}'.format(info_path))


if __name__ == '__main__':
  main()
