import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

#定义了一个名为 BasePolicy 的PyTorch神经网络模型
class BasePolicy(nn.Module):
    """
    基础策略网络
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim（int）: 输入特征的维度
            out_dim（int）: 输出特征的维度
            hidden_dim（int，默认为64）: 隐藏层的维度
            nonlin（PyTorch激活函数，默认为F.leaky_relu）: 隐藏层的非线性激活函数
            norm_in（布尔值，默认为True）: 是否对输入进行批量归一化
            onehot_dim（int，默认为0）: 如果有one-hot编码的特征，指定其维度
        """
        super(BasePolicy, self).__init__() #初始化nn的基本功能

        if norm_in:#如果norm_in为True，创建一个nn.BatchNorm1d层，用于对输入进行批量归一化
            #类的实例就拥有了一个一维批归一化层，可以在后续的代码中使用该层对输入进行归一化操作
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            #self.in_fn 被赋值为 lambda x: x，其中 x 是输入参数。
            #lambda 函数的主体部分是 x，它直接返回输入值 x，因此该函数实现了一个恒等映射
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    #前向传播函数
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): 观测值批次（可选，还包括 onehot 标签的元组）
        Outputs:
            out (PyTorch Matrix): 行为
        """
        # 如果X是元组，将X的第一个元素（通常是观察值）赋值给X
        # 并将元组的第二个元素（通常是one-hot编码）赋值给onehot
        # X不是元组，则onehot保持为None
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  #在神经网络中用于归一化输入
        if onehot is not None:
            #如果存在one-hot编码特征，将其与inp拼接在一起，形成新的输入。
            # 在这里，torch.cat函数在第二维度（dim=1）上拼接输入
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy): #离散策略
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
