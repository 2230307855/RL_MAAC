import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau) 执行 DDPG 软更新（根据权重因子 tau 将目标参数移向源）
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update 更新的权重因子
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source): #将参数从源网络转移到目标网络
    """
    Copy network parameters from source to target 将网络参数从源复制到目标
    Inputs:
        target (torch.nn.Module): Net to copy parameters to 要将参数复制到的网络
        source (torch.nn.Module): Net whose parameters to copy 要复制参数的网络
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data) #直接复制源网络的参数

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

#实现ε-greedy策略
#在选择动作时综合了贪婪（greedy）和探索
#根据给定的logits（通常是一个代表动作值的向量）生成一个one-hot编码的动作向量
def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy 给定一批 logit，使用 epsilon 贪婪策略返回 one-hot sample
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    #找到在当前策略下最佳的动作，即logits中概率最大的动作，然后将其转换为one-hot形式
    #在给定维度（默认为1）上计算 logits 张量的最大值
    #即将概率最大的动作位置设置为1，其他位置设置为0
    #float() 是一个类型转换操作，将布尔型张量转换为浮点型张量，将 True 转换为1.0，False 转换为0.0
    #argmax_acs 是一个浮点型张量，其形状与 logits 相同，其中最大值所在位置的元素为1.0，其他位置的元素为0.0
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0: #ε的值为0，表示完全按照贪婪策略选择动作，直接返回上一步得到的one-hot编码的动作
        return argmax_acs
    #对于每一行，它随机选择一个动作的索引。
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    #使用 epsilon greedy在最佳操作和随机操作之间进行选择
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=dim)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y

def firmmax_sample(logits, temperature, dim=1):
    if temperature == 0:
        return F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)) / temperature
    return F.softmax(y, dim=dim)

#从一个给定的概率分布 probs 中进行多项式采样
#函数将根据这个概率分布进行采样，返回采样得到的动作的索引和对应的 one-hot 编码
def categorical_sample(probs, use_cuda=False):
    #使用 multinomial 函数从 probs 中进行采样，每次只采样一个动作
    #返回一个包含采样的动作索引的张量 int_acs
    int_acs = torch.multinomial(probs, 1)
    #判断是否需要在 GPU 上计算
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    #创建一个与 probs 同样形状的全零张量，并将其封装为Variable对象。
    #这个张量将用于存储 one-hot 编码的动作
    #使用 scatter_ 函数将 int_acs 中对应位置的值置为1，得到 one-hot 编码的动作
    #scatter_(维度，索引，填充值) 哪个维度，哪个位置，撒什么点
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

def sep_clip_grad_norm(parameters, max_norm, norm_type=2):
    """
    Clips gradient norms calculated on a per-parameter basis, rather than over
    the whole list of parameters as in torch.nn.utils.clip_grad_norm.
    Code based on torch.nn.utils.clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        if norm_type == float('inf'):
            p_norm = p.grad.data.abs().max()
        else:
            p_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (p_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)
