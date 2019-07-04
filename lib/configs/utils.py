import time
import torch
import torch.nn as nn


def calculate_loss(c_logits, f_logits, c_ys, f_ys, c2f, f2c):
  # bs, n_c; bs, n_f
  batch_size = c_logits.size(0)
  select_x  = [bs for bs in range(0, batch_size)]
  select_cy = c_ys.tolist()
  select_fy = f_ys.tolist()
  c_max, _ = torch.max(c_logits, dim=1)
  c_logit  = c_logits[select_x, select_cy] 
  f_logit  = f_logits[select_x, select_fy] 
  
  loss_c   = torch.sum(-(c_logit-c_max)) + torch.sum(torch.log(torch.sum(torch.exp(c_logits - torch.unsqueeze(c_max, 1)), dim=1)), dim=0) 
  loss_f_lst   = []
  for idx, f_y in enumerate(f_ys):
    f_candidates = c2f[c_ys[idx].item()] 
    f_can_logit = f_logits[idx, f_candidates]
    f_max  = torch.max(f_can_logit)
    loss_f1  = -(f_logit[idx] - f_max) 
    loss_f2  = torch.log(torch.sum(torch.exp(f_can_logit - f_max)))
    loss_f_lst.append(loss_f1+loss_f2)
  loss_f = sum(loss_f_lst)
  loss = (loss_c+loss_f)/batch_size
  return loss 

def calculate_acc(c_logits, f_logits, c_ys, f_ys, c2f_list, fy2cy):
  coarse_pred = nn.functional.softmax(c_logits, dim=1)
  predictions = [None for _ in range(f_logits.size(1))]
  for cls, finelist in c2f_list.items():
    coarse_P    = coarse_pred[:, cls]
    same_coarse_logits = f_logits[:, finelist]
    same_coarse_predcs = nn.functional.softmax(same_coarse_logits, dim=1)
    for idx, finecls in enumerate(finelist):
      predictions[finecls] = same_coarse_predcs[:, idx] * coarse_P
  predictions = torch.stack(predictions, -1)
  #loss = nn.functional.nll_loss(torch.log(predictions), f_ys) 
  return coarse_pred, predictions 

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count


def obtain_accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # bs*k
    pred = pred.t()  # t: transpose, k*bs
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 1*bs --> k*bs

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def convert_secs2time(epoch_time, string=True):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if string:
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return need_time
  else:
    return need_hour, need_mins, need_secs

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d-%X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def reverse_dict(ori_dict_list):
  rev_dict = {}
  for k,v in ori_dict_list.items():
    for v_i in v:
      if not v_i in rev_dict:
        rev_dict[v_i] = k
  return rev_dict
