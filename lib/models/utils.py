import torch
from configs import obtain_accuracy

def euclidean_dist(x, y, transform=True):
  bs = x.shape[0]
  if transform:
    num_proto = y.shape[0]
    query_lst = []
    for i in range(bs):
      ext_query = x[i, :].repeat(num_proto, 1)
      query_lst.append(ext_query)
    x = torch.cat(query_lst, dim=0)
    y = y.repeat(bs, 1)
 
  return torch.pow(x - y, 2).sum(-1)

def update_acc(coef_lst, proto_lst, acc_lst, lvl_imgs_emb, query_idx, lvl_meta_labels):
  for coef, proto, acc in zip(coef_lst, proto_lst, acc_lst):
    if coef > 0:
      n_query = len(query_idx)
      logits = - euclidean_dist(lvl_imgs_emb, proto, transform=True).view(n_query, len(proto))
      top_fs = obtain_accuracy(logits, lvl_meta_labels.data, (1,))
      acc.update(top_fs[0].item(), n_query) 


def get_proto_ss(all_level_classes, ancestors, cpu_wordids, lvls_embs, proto_base, avg_balance_coef, temp):
  proto_lst, proto_ss_lst, anc_idx_lst, emb_lst = [], [], [], [] 
  for i, cls in enumerate(all_level_classes):
    ancs = [a for a in ancestors[cls]]
    ancs_idx  = [idx for idx, wordid in enumerate(cpu_wordids) if wordid in ancs] 
    anc_idx_lst.append(ancs_idx) 
    num_anc_idx = len(ancs_idx)
    if num_anc_idx > 0:
      emb = lvls_embs[torch.tensor(ancs_idx).long()]
      emb_lst.append( emb )
      proto_lst.append(proto_base[i].unsqueeze(0).repeat(num_anc_idx, 1))
  # TODO: check if other levels except lvl3 has no ancestors 
  # ancestor and descendants info can be predefined
  anc_emb   = torch.cat(emb_lst, dim=0)
  proto_cur = torch.cat(proto_lst, dim=0) 
  knn_dis   = euclidean_dist(anc_emb, proto_cur, transform=False)
  start_idx = 0
  for i, cls in enumerate(all_level_classes): 
    len_idx = len(anc_idx_lst[i])
    if len_idx == 0:
      proto_ss_lst.append(proto_base[i])
    else:
      k_dis = knn_dis[start_idx: start_idx+len_idx]
      a_emb = anc_emb[start_idx: start_idx+len_idx]
      p_ss_weight = torch.nn.functional.softmax((-k_dis + avg_balance_coef)/temp, dim=0)
      proto_ss_lst.append(( a_emb * p_ss_weight.unsqueeze(1) ).mean(0))
      start_idx = start_idx + len_idx 
  proto_ss = torch.stack(proto_ss_lst, dim=0) 
  return proto_ss

def get_proto_att_ch(all_level_classes, children, proto_base, att_model):
  # proto_att_ch
  proto_att_ch_lst, num_chil_lst, proto_att_lst, chil_lst = [], [], [], []
  for idx, cls in enumerate(all_level_classes):
    chil_idx = [all_level_classes.index(ch) for ch in children[cls] if ch in all_level_classes]
    num_chil = len(chil_idx)
    num_chil_lst.append(num_chil)
    if num_chil > 0:
      proto_cur = proto_base[idx].unsqueeze(0).repeat(num_chil, 1)
      proto_att_lst.append(proto_cur)
      chil_proto = proto_base[torch.tensor(chil_idx).long()]
      chil_lst.append(chil_proto)
  proto_cur = torch.cat(proto_att_lst, dim=0)
  proto_chi = torch.cat(chil_lst, dim=0)

  raw_att_score = att_model( proto_chi, proto_cur ) 
  
  start_idx = 0 
  for i, cls in enumerate(all_level_classes):
    num_chil = num_chil_lst[i]
    if num_chil == 0:
      proto_att_ch_lst.append(proto_base[i])
    else:
      raw = raw_att_score[start_idx: start_idx + num_chil] 
      att_score   = torch.nn.functional.softmax(raw, dim=0) 
      ch_proto    = proto_chi[start_idx: start_idx + num_chil] 
      proto_att_ch_lst.append( torch.sum( att_score.unsqueeze(1) * ch_proto, dim=0) )
      start_idx = start_idx + num_chil
  proto_att_ch = torch.stack(proto_att_ch_lst, dim=0)
  return proto_att_ch


def get_att_proto(all_level_classes, neighbours, proto_base, att_model, n_hop=1):
  for n_h in range(n_hop):
    # proto_att_ch
    proto_att_nei_lst, num_nei_lst, proto_att_lst, nei_lst = [], [], [], []
    for idx, cls in enumerate(all_level_classes):
      nei_idx = [all_level_classes.index(ch) for ch in neighbours[cls] if ch in all_level_classes]
      num_nei = len(nei_idx)
      num_nei_lst.append(num_nei)
      if num_nei > 0:
        proto_cur = proto_base[idx].unsqueeze(0).repeat(num_nei, 1)
        proto_att_lst.append(proto_cur)
        nei_proto = proto_base[torch.tensor(nei_idx).long()]
        nei_lst.append(nei_proto)
    proto_cur = torch.cat(proto_att_lst, dim=0)
    proto_chi = torch.cat(nei_lst, dim=0)

    raw_att_score = att_model( proto_chi, proto_cur ) 
    
    start_idx = 0 
    for i, cls in enumerate(all_level_classes):
      num_nei = num_nei_lst[i]
      if num_nei == 0:
        proto_att_nei_lst.append(proto_base[i])
      else:
        raw = raw_att_score[start_idx: start_idx + num_nei] 
        att_score = torch.nn.functional.softmax(raw, dim=0) 
        ch_proto  = proto_chi[start_idx: start_idx + num_nei] 
        proto_att_nei_lst.append( torch.sum( att_score.unsqueeze(1) * ch_proto, dim=0) )
        start_idx = start_idx + num_nei
    proto_att_ch = torch.stack(proto_att_nei_lst, dim=0)
    proto_base   = proto_att_ch
    
  return proto_att_ch
