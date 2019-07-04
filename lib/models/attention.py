import torch
import torch.nn as nn

class Attention_cosine(nn.Module):
  def __init__(self, input_dim):
    super(Attention_cosine, self).__init__()
    #dropout_ratio   = 0
    self.linear_q = nn.Linear(input_dim, 128, bias=False)
    self.linear_k = nn.Linear(input_dim, 128, bias=False)
    self.cosine   = nn.CosineSimilarity(dim=-1, eps=1e-4)

    # module initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        #nn.init.kaiming_normal_(m.weight)
        m.weight.data.normal_(0, 0.001)
        #m.bias.data.zero_()

  def forward(self, children_proto, proto):
    '''
    - children_proto: n_children, f_d  
    - proto:   1, f_d 
    '''
    k = self.linear_k(children_proto) #  bs, f_d
    q = self.linear_q(proto)
    raw_att_score = self.cosine(q,k)
    return raw_att_score
