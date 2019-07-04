import time
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from configs import AverageMeter, obtain_accuracy, time_string
from models import euclidean_dist, update_acc, get_att_proto

def get_test_proto(few_shot_pp, proto_lst):
  with torch.no_grad():
    new_pp_lst = []
    ori_proto_lst = deepcopy(proto_lst)
    for pp in few_shot_pp: 
      proto_lst.append(pp)
      pp = pp.view(1,-1)
      candidates = torch.stack(proto_lst, dim=0)
      dis = - torch.pow(pp-candidates, 2).sum(-1)
      sim = torch.nn.functional.softmax(dis, 0).view(-1, 1)
      new_pp = torch.sum( candidates * sim, dim=0 )
      new_pp_lst.append(new_pp)
      proto_lst = ori_proto_lst
    final_proto = torch.stack(new_pp_lst, dim=0)
  return final_proto

def get_test_proto_v2(few_shot_pp, proto_lst):
  with torch.no_grad():
    new_pp_lst = []
    ori_proto_lst = deepcopy(proto_lst)
    n_nei = 8
    for pp in few_shot_pp: 
      pp = pp.view(1,-1)
      candidates = torch.stack(proto_lst, dim=0)
      dis = - torch.pow(pp-candidates, 2).sum(-1)
      n_nei = candidates.shape[0] # uncomment this to make that work
      top_dis, top_idx = torch.topk(dis, n_nei)
      top_can = candidates[top_idx]
      top_sim = torch.nn.functional.softmax(top_dis, 0).view(-1, 1)
      new_pp = torch.sum( top_can * top_sim, dim=0 )
      new_pp_lst.append(new_pp)
      proto_lst = ori_proto_lst
    final_proto = torch.stack(new_pp_lst, dim=0)
  return final_proto

# different par_ratio 
def get_test_proto_v3(few_shot_pp, proto_lst, par_ratio):
  with torch.no_grad():
    new_pp_lst = []
    ori_proto_lst = deepcopy(proto_lst)
    n_nei = 8
    for pp in few_shot_pp: 
      pp = pp.view(1,-1)
      candidates = torch.stack(proto_lst, dim=0)
      dis = - torch.pow(pp-candidates, 2).sum(-1)
      n_nei = candidates.shape[0] # uncomment this to make that work
      top_dis, top_idx = torch.topk(dis, n_nei)
      top_can = candidates[top_idx]
      top_sim = torch.nn.functional.softmax(top_dis, 0).view(-1, 1)
      top_sim = torch.cat( (torch.tensor( (1-par_ratio)).cuda().view(1,1), top_sim*par_ratio), dim=0 )
      top_can = torch.cat((pp, top_can), dim=0) 
      new_pp = torch.sum( top_can * top_sim, dim=0 )
      new_pp_lst.append(new_pp)
      proto_lst = ori_proto_lst
    final_proto = torch.stack(new_pp_lst, dim=0)
  return final_proto

def get_att_proto(few_shot_pp, candidates, prop_proto, att_model, par_ratio, n_nei):
  with torch.no_grad():
    new_pp_lst = []
    #n_nei = candidates.shape[0]
    for pp in few_shot_pp: 
      pp = pp.view(1,-1)
      dis = - torch.pow(pp-prop_proto, 2).sum(-1)
      #n_nei = candidates.shape[0] # uncomment this to make that work
      top_dis, top_idx = torch.topk(dis, n_nei)
      top_can = candidates[top_idx]
      proto_cur = pp.repeat(n_nei, 1)
      raw_att_score = att_model( top_can, proto_cur ) 
      top_sim = torch.nn.functional.softmax(raw_att_score, dim=0) 
      top_sim = torch.cat((torch.tensor( (1-par_ratio)).cuda().view(1,1), (top_sim*par_ratio).view(-1,1)), dim=0 )
      top_can = torch.cat((pp, top_can), dim=0) 
      new_pp = torch.sum( top_can * top_sim, dim=0 )
      new_pp_lst.append(new_pp)
    final_proto = torch.stack(new_pp_lst, dim=0)
  return final_proto

def get_att_proto_vote(lvls_embs, grouped_s_idxs, few_shot_pp, candidates, prop_proto, att_model, par_ratio, n_nei):
  with torch.no_grad():
    new_pp_lst = []
    for idx, s_idx in enumerate(grouped_s_idxs):
      pp = few_shot_pp[idx].view(1, -1)
      s_idx_lst = list(s_idx)
      vote_lst = []
      for s_idx in s_idx_lst:
        img_fea = lvls_embs[s_idx.item()].view(1,-1)
        dis = - torch.pow(img_fea-prop_proto, 2).sum(-1) 
        vote_lst.append(dis)
      votes = sum( vote_lst )
      top_dis, top_idx = torch.topk(votes, n_nei)
      top_can = candidates[top_idx]
      proto_cur = pp.repeat(n_nei, 1)
      raw_att_score = att_model( top_can, proto_cur )
      top_sim = torch.nn.functional.softmax(raw_att_score, dim=0)
      top_sim = torch.cat((torch.tensor( (1-par_ratio)).cuda().view(1,1), (top_sim*par_ratio).view(-1,1)), dim=0 )
      top_can = torch.cat((pp, top_can), dim=0)
      new_pp = torch.sum( top_can * top_sim, dim=0 )
      new_pp_lst.append(new_pp)
    final_proto = torch.stack(new_pp_lst, dim=0) 
  return final_proto 


def test_no_hierarchy(lr_scheduler, par_ratio, emb_model, att_par, att_chi, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support, pp_buffer, prop_proto):
  
  args = deepcopy(args)
  ancestors, parents, descendants, children = hierarchy_info
  losses, acc1 = AverageMeter(), AverageMeter()
  acc_base, acc_par, acc_chi = AverageMeter(), AverageMeter(), AverageMeter()
  data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
  num_device = len(emb_model.device_ids)

  par_ratio = par_ratio / 10
  #logger.print("par_ratio is {}".format(par_ratio))

  if mode == "test":
    emb_model.eval(); att_par.eval(); att_chi.eval()
    metaval_accuracies = []
  else: raise TypeError("invalid mode {:}".format(mode))

  for batch_idx, (img_idx, imgs, labels, levels, wordids) in enumerate(dataloader):
    cpu_levels        = levels.tolist()
    cpu_wordids       = wordids.tolist()
    all_levels        = list(set(cpu_levels)); all_levels.sort()

    lvls_embs  = emb_model(imgs)

    lvls_wordids = []
    # get idx, label, meta_label for every level
    for lvl in all_levels:
      lvl_wordids    = sorted(set([wordid for level, wordid in zip(cpu_levels, cpu_wordids) if level == lvl]))
      lvls_wordids.append(lvl_wordids)
      if lvl == max(all_levels):
        lvl_idx = [idx for idx, wd in enumerate(cpu_wordids) if wd in lvl_wordids]
        lvl_labels_dicts = { x:i for i, x in enumerate(lvl_wordids)}

        grouped_s_idxs, grouped_q_idxs, few_shot_pp = [], [], []
        idxs_dict = defaultdict(list)
        for i in lvl_idx:
          wordid = cpu_wordids[i]
          idxs_dict[wordid].append(i)
        idxs_dict = dict(sorted(idxs_dict.items()))
        for wordid, idxs in idxs_dict.items():
          s_idxs = idxs[:n_support]
          grouped_s_idxs.append(torch.IntTensor(s_idxs))
          grouped_q_idxs.append(torch.IntTensor(idxs[n_support:]))
          few_shot_pp.append( torch.mean(lvls_embs[s_idxs], dim=0) )
        support_idxs = torch.cat(grouped_s_idxs, dim=0).tolist()
        query_idxs   = torch.cat(grouped_q_idxs, dim=0).tolist()
        lvl_meta_labels  = torch.LongTensor( [lvl_labels_dicts[cpu_wordids[x]] for x in query_idxs] )
        lvl_meta_labels  = lvl_meta_labels.cuda(non_blocking=True)
        # lvls_wordids cls_support_idxs cls_query_idxs all_level_classes are in the same order, low--high level, in each level, small--big      
    all_level_classes  = [c for sublist in lvls_wordids for c in sublist] 

    '''
    if pp_buffer:
      if mode =="test":
        proto_lst = []
        for idx, cls in enumerate(all_level_classes):
          if torch.sum( pp_buffer.pp_running[cls] ).item() > 0: # common classes
            proto_lst.append(pp_buffer.pp_running[cls])
      else: raise ValueError("invalid mode {}".format(mode))
    else:
      raise ValueError("invalid : no buffer ")
    '''
    candidates = pp_buffer.pp_running
    voting = False
    #voting = True 
    n_nei = 3
    # classification over every level
    loss_lvls = []
    for i, lvl in enumerate(all_levels): 
      if lvl == max(all_levels):
        if voting:
          final_proto = get_att_proto_vote(lvls_embs, grouped_s_idxs, few_shot_pp, candidates, prop_proto, att_par, par_ratio, n_nei)
        else:
          final_proto  = get_att_proto(few_shot_pp, candidates, prop_proto, att_par, par_ratio, n_nei)
        lvl_imgs_emb = lvls_embs[query_idxs] 
        logits       = - euclidean_dist(lvl_imgs_emb, final_proto, transform=True).view(len(query_idxs), len(final_proto))
        loss         = criterion(logits, lvl_meta_labels)
        loss_lvls.append(loss)
        top_fs        = obtain_accuracy(logits, lvl_meta_labels.data, (1,))
        acc1.update(top_fs[0].item(), len(query_idxs))
    loss = sum(loss_lvls)
    losses.update(loss.item(), len(query_idxs))  

    if mode == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    elif mode=="test":
      metaval_accuracies.append(top_fs[0].item())
      if batch_idx + 1 == len(dataloader):
        metaval_accuracies = np.array(metaval_accuracies)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(batch_idx + 1)
        #logger.print("ci95 is : {:}".format(ci95))
    else: raise ValueError('Invalid mode = {:}'.format( mode ))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    '''
    if (mode=="train" and ((batch_idx % args.log_interval == 0) or (batch_idx + 1 == len(dataloader)))) \
    or (mode=="test" and (batch_idx + 1 == len(dataloader))):
      Tstring = 'TIME[{data_time.val:.2f} ({data_time.avg:.2f}) {batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(data_time=data_time, batch_time=batch_time)
      Sstring = '{:} {:} [Epoch={:03d}/{:03d}] [{:03d}/{:03d}]'.format(time_string(), mode, epoch, args.epochs, batch_idx, len(dataloader))
      Astring = 'loss=({:.3f}, {:.3f}), loss_lvls:{}, loss_min:{:.2f}, loss_max:{:.2f}, loss_mean:{:.2f}, loss_var:{:.2f}, acc@1=({:.2f}, {:.2f}), acc@base=({:.1f}, {:.1f}), acc@par=({:.1f}, {:.1f}), acc@chi=({:.1f}, {:.1f})'.format(losses.val, losses.avg, [l.item() for l in loss_lvls], min(loss_lvls).item(), max(loss_lvls).item(), torch.mean(torch.stack(loss_lvls)).item(), torch.var(torch.stack(loss_lvls)).item(), acc1.val, acc1.avg, acc_base.val, acc_base.avg, acc_par.val, acc_par.avg, acc_chi.val, acc_chi.avg)
      Cstring = 'p_base_weigth : {:.4f}; p_par_weight : {:.4f}; p_chi_weight : {:.4f}'.format(args.coef_base, args.coef_anc, args.coef_chi)

      logger.print('{:} {:} {:} {:} \n'.format(Sstring, Tstring, Astring, Cstring))
    '''
  return losses, acc1, ci95
