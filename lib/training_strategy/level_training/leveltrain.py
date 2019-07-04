import time
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from configs import AverageMeter, obtain_accuracy, time_string
from models import euclidean_dist, update_acc, get_att_proto

def leveltrain(lr_scheduler, emb_model, att_par, att_chi, criterion, optimizer, logger, dataloader, hierarchy_info, wordid_level_label, epoch, args, mode, n_support, pp_buffer=None):
  
  args = deepcopy(args)
  ancestors, parents, descendants, children = hierarchy_info
  losses, acc1 = AverageMeter(), AverageMeter()
  acc_base, acc_par, acc_chi = AverageMeter(), AverageMeter(), AverageMeter()
  data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
  num_device = len(emb_model.device_ids)

  if mode == "train":
    emb_model.train(); att_par.train(); att_chi.train()
  elif mode == "test":
    emb_model.eval(); att_par.eval(); att_chi.eval()
    metaval_accuracies = []
  else: raise TypeError("invalid mode {:}".format(mode))

  for batch_idx, (img_idx, imgs, labels, levels, wordids) in enumerate(dataloader):
    if mode=="train" and lr_scheduler:
      lr_scheduler.step()
    cpu_levels        = levels.tolist()
    cpu_wordids       = wordids.tolist()
    all_levels        = list(set(cpu_levels)); all_levels.sort()

    # get idx, label, meta_label for every level
    lvls_wordids, lvls_meta_labels, cls_support_idxs, cls_query_idxs = [], [], [], []
    for lvl in all_levels:
      lvl_wordids    = sorted(set([wordid for level, wordid in zip(cpu_levels, cpu_wordids) if level == lvl]))
      lvl_idx = [idx for idx, wd in enumerate(cpu_wordids) if wd in lvl_wordids]
      lvl_labels_dicts = { x:i for i, x in enumerate(lvl_wordids)}

      grouped_s_idxs, grouped_q_idxs = [], []
      idxs_dict = defaultdict(list)
      for i in lvl_idx:
        wordid = cpu_wordids[i]
        idxs_dict[wordid].append(i)
      idxs_dict = dict(sorted(idxs_dict.items()))
      # for non-few-shot classes, support and query set are the same
      if lvl < max(all_levels):
        for wordid, idxs in idxs_dict.items():
          grouped_s_idxs.append(torch.IntTensor(idxs))
        grouped_q_idxs = grouped_s_idxs
      else:
        for wordid, idxs in idxs_dict.items():
          grouped_s_idxs.append(torch.IntTensor(idxs[:n_support]))
          grouped_q_idxs.append(torch.IntTensor(idxs[n_support:]))
      support_idxs = torch.cat(grouped_s_idxs, dim=0).tolist()
      query_idxs   = torch.cat(grouped_q_idxs, dim=0).tolist()
      lvl_meta_labels  = torch.LongTensor( [lvl_labels_dicts[cpu_wordids[x]] for x in query_idxs] )
      lvl_meta_labels  = lvl_meta_labels.cuda(non_blocking=True)
      # lvls_wordids cls_support_idxs cls_query_idxs all_level_classes are in the same order, low--high level, in each level, small--big      
      lvls_wordids.append(lvl_wordids); lvls_meta_labels.append(lvl_meta_labels)
      cls_support_idxs.extend(grouped_s_idxs); cls_query_idxs.extend(grouped_q_idxs)
    all_level_classes  = [c for sublist in lvls_wordids for c in sublist]

    lvls_embs  = emb_model(imgs)
    if pp_buffer:
      if mode == "train":
        proto_base = pp_buffer.pp_running[all_level_classes] 
      elif mode =="test":
        proto_base_lst = []
        for idx, cls in enumerate(all_level_classes):
          if torch.sum( pp_buffer.pp_running[cls] ).item() > 0: # common classes  
            s_i = cls_support_idxs[idx]
            pp_new = 0.5 * pp_buffer.pp_running[cls] + 0.5 * torch.mean(lvls_embs[s_i.long()], dim=0)
            proto_base_lst.append(pp_new)
          else: # non-common classes
            s_i = cls_support_idxs[idx]
            proto_base_lst.append(torch.mean(lvls_embs[s_i.long()], dim=0) )
        proto_base = torch.stack(proto_base_lst, dim=0)
      else: raise ValueError("invalid mode {}".format(mode))
    else:
      proto_base_lst = [torch.mean(lvls_embs[s_i.long()], dim=0) for s_i in cls_support_idxs]   
      proto_base = torch.stack(proto_base_lst, dim=0)

    if not args.coef_anc:
      proto_par = 0
    else:
      proto_par = get_att_proto(all_level_classes, parents, proto_base, att_par, args.n_hop)
    if not args.coef_chi:
      proto_chi = 0
    else:
      proto_chi = get_att_proto(all_level_classes, children, proto_base, att_chi, args.n_hop)
    
    coef_lst  = [args.coef_base, args.coef_anc, args.coef_chi]
    proto_lst = [proto_base, proto_par, proto_chi]
    acc_lst   = [acc_base, acc_par, acc_chi]
    # p_final
    if "avg" in args.training_strategy:
      denominator = args.coef_base + args.coef_anc + args.coef_chi 
      final_proto = (args.coef_base/denominator) * proto_base + (args.coef_anc/denominator) * proto_par + (args.coef_chi/denominator) * proto_chi
    elif "weighted" in args.training_strategy:
      # TODO: hardcode
      final_proto = 0.3 * proto_base + 0.7 * proto_par 
      #final_proto = 0.45 * proto_base + 0.45 * proto_par + 0.1 * proto_chi
    elif "relation" in args.training_strategy:
      cat_proto = torch.cat( [proto for coef, proto in zip(coef_lst, proto_lst) if coef > 0], dim=-1)
      final_proto = relation_nn(cat_proto)
    else:
      raise ValueError("undefined training_strategy {}, no proto_weight info inside".format(args.training_strategy))

    # classification over every level
    loss_lvls = []
    for i, lvl in enumerate(all_levels): 
      lvl_wordid = lvls_wordids[i] 
      idx_start  = all_level_classes.index(lvl_wordid[0])
      idx_end    = all_level_classes.index(lvl_wordid[-1]) 
      protos     = final_proto[idx_start:idx_end+1]
      query_idx  = torch.cat( cls_query_idxs[idx_start:idx_end+1], dim=0 ).long()
      lvl_imgs_emb = lvls_embs[query_idx] 
      lvl_meta_labels = lvls_meta_labels[i]
      logits       = - euclidean_dist(lvl_imgs_emb, protos, transform=True).view(len(query_idx), len(protos))
      loss         = criterion(logits, lvl_meta_labels)
      loss_lvls.append(loss)
      if lvl == max(all_levels):
        top_fs        = obtain_accuracy(logits, lvl_meta_labels.data, (1,))
        acc1.update(top_fs[0].item(), len(query_idx))
        fine_proto_lst = []
        for c,p in zip( coef_lst, proto_lst ):
          if c>0:
            fine_proto_lst.append( p[idx_start:idx_end+1] )
          else:
            fine_proto_lst.append( 0 )
        update_acc(coef_lst, fine_proto_lst, acc_lst, lvl_imgs_emb, query_idx, lvl_meta_labels) 
      if "multi" in args.training_strategy:
        for coef, proto in zip(coef_lst, proto_lst):
          if coef > 0:
            logits = - euclidean_dist(lvl_imgs_emb, proto, transform=True).view(len(query_idx), len(proto))
            loss   = criterion(logits, lvl_meta_labels)
            loss_lvls.append(loss)
    if "mean" in args.training_strategy:
      loss = sum(loss_lvls) / len(loss_lvls)
    elif "single" in args.training_strategy or "multi" in args.training_strategy:
      loss = sum(loss_lvls)
    elif "selective" in args.training_strategy:
      loss = sum(loss_lvls[:-1])/10 + loss_lvls[-1]
    else: raise ValueError("undefined loss type info in training_strategy : {}".format(args.training_strategy))
      
    losses.update(loss.item(), len(query_idx))  

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
        logger.print("ci95 is : {:}".format(ci95))
    else: raise ValueError('Invalid mode = {:}'.format( mode ))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if (mode=="train" and ((batch_idx % args.log_interval == 0) or (batch_idx + 1 == len(dataloader)))) \
    or (mode=="test" and (batch_idx + 1 == len(dataloader))):
      Tstring = 'TIME[{data_time.val:.2f} ({data_time.avg:.2f}) {batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(data_time=data_time, batch_time=batch_time)
      Sstring = '{:} {:} [Epoch={:03d}/{:03d}] [{:03d}/{:03d}]'.format(time_string(), mode, epoch, args.epochs, batch_idx, len(dataloader))
      Astring = 'loss=({:.3f}, {:.3f}), loss_lvls:{}, loss_min:{:.2f}, loss_max:{:.2f}, loss_mean:{:.2f}, loss_var:{:.2f}, acc@1=({:.1f}, {:.1f}), acc@base=({:.1f}, {:.1f}), acc@par=({:.1f}, {:.1f}), acc@chi=({:.1f}, {:.1f})'.format(losses.val, losses.avg, [l.item() for l in loss_lvls], min(loss_lvls).item(), max(loss_lvls).item(), torch.mean(torch.stack(loss_lvls)).item(), torch.var(torch.stack(loss_lvls)).item(), acc1.val, acc1.avg, acc_base.val, acc_base.avg, acc_par.val, acc_par.avg, acc_chi.val, acc_chi.avg)
      Cstring = 'p_base_weigth : {:.4f}; p_par_weight : {:.4f}; p_chi_weight : {:.4f}'.format(args.coef_base, args.coef_anc, args.coef_chi)

      logger.print('{:} {:} {:} {:} \n'.format(Sstring, Tstring, Astring, Cstring))
  return losses, acc1, acc_base, acc_par, acc_chi
