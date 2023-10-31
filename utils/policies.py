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
    离散行动空间策略网络
    """
    #接受任意数量的位置参数（*args）和关键字参数（**kwargs）
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)
    """
    obs: 代表输入的观察值（observations）或状态（states）。这是策略网络的输入，用于决定输出的动作
    sample=True: 一个布尔值参数，表示是否使用采样策略
    return_all_probs=False: 一个布尔值参数，表示是否返回所有可能动作的概率分布。
                            如果为True，策略网络将会返回所有动作的概率分布.
    return_log_pi=False: 一个布尔值参数，表示是否返回选择动作的对数概率。
                         如果为True，策略网络将会返回选择的动作的对数概率。
    regularize=False: 一个布尔值参数，表示是否对策略网络进行正则化。
                         如果为True，策略网络将会返回正则化项.
    return_entropy=False: 一个布尔值参数，表示是否返回策略网络的熵(概率分布的混乱程度度量)
    """
    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        #
        probs = F.softmax(out, dim=1)
        # 检查模型参数是否在GPU上，用于后续操作的设备选择
        on_gpu = next(self.parameters()).is_cuda
        # 如果sample参数为True，表示需要从概率分布中采样动作
        if sample: # ！！！
            # 函数返回采样的动作的索引（int_act）和one-hot编码的动作（act）
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            # 使用onehot_from_logits函数，将概率分布probs转换为one-hot编码的动作
            act = onehot_from_logits(probs)
        # 创建一个包含动作信息的列表rets，初始时只包含采样的动作或贪婪策略选择的动作
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1) # 计算输出的对数概率分布
        if return_all_probs:
            rets.append(probs) # 将所有动作的概率分布添加到rets列表
        if return_log_pi:
            # return log probability of selected action
            # ！！！
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            #计算输出的平方的均值，通常作为正则化项添加到rets列表中
            rets.append([(out**2).mean()])
        if return_entropy:
            #计算熵并将其添加到rets列表中
            #取所有行的熵值的平均值，得到整个概率分布的熵
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
