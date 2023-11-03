import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

# 定义了一个Attention Critic模型，用作所有智能体的评论者（critic）
# 允许每个智能体根据其他智能体的观测和动作进行注意力权重的计算,评估该智能体的动作的质量
class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): sa_sizes 是一个包含每个智能体状态和动作空间大小的列表
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): 表示是否对输入应用Batch Normalization
            attend_heads (int): 注意力头的数量，用于多头注意力机制
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList() # 包含每个智能体的观测和动作编码器
        '''
        nn.ModuleList 用于封装多个nn.Module子模块的列表
            提供了一种方便的方式来管理和迭代访问多个子模块
            子模块可以是任何继承自nn.Module的类
            自动注册和管理其包含的子模块的参数
            调用parameters()方法时，会返回所有子模块的参数，从而使得整个模型的参数可以方便地进行优化
            自动调用其包含的子模块的前向传播方法
            通过索引或迭代的方式访问nn.ModuleList中的子模块
            子模块需要在模型的构造函数中进行显式定义和初始化，以便正确注册和管理参数
        '''
        self.critics = nn.ModuleList() # 含每个智能体的评论者

        self.state_encoders = nn.ModuleList() # 每个智能体的状态编码器
        # iterate over
        # Attention Critic模型的编码器
        # 三个神经网络模块：观测和动作的编码器（encoder），评论者（critic），以及状态的编码器（state_encoder）
        for sdim, adim in sa_sizes: # 每个智能体的状态维度（sdim）和动作维度（adim）
            idim = sdim + adim # 计算输入维度（idim）和输出维度（odim）
            odim = adim


            # 创建一个Sequential容器，用于构建观测和动作的编码器。
            encoder = nn.Sequential()
            if norm_in: # 如果norm_in为True，添加Batch Normalization层，用于对输入进行规范化
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            # 全连接层（线性变换），将输入维度映射到隐藏层维度
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            # 将观测和动作的编码器添加到self.critic_encoders列表中，用于每个智能体
            self.critic_encoders.append(encoder)


            #  创建一个Sequential容器，用于构建评论者
            critic = nn.Sequential()
            # 添加一个全连接层，将两倍隐藏层维度的输入映射到隐藏层维度
            '''
            2 * hidden_dim 表示输入层的维度是两倍的隐藏层维度。
            在这个模型中，评论者（critic）的输入包括两部分信息：
                智能体的观测（states）和其他智能体的观测和动作编码（other_all_values）。
            因此，将这两部分信息连接起来形成一个更大的向量，然后通过一个线性变换将其映射到隐藏层维度。
            这种设计允许评论者网络同时考虑当前智能体的观测和其他智能体的编码信息，以更好地估计Q值。
            这种联合表示可以帮助智能体学习到团队策略，而不仅仅是个体策略
            '''
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                    hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            # 全连接层，将隐藏层维度的输入映射到输出维度（评论者的输出维度）
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)


            # 构建状态的编码器
            state_encoder = nn.Sequential()
            if norm_in: # 用于对agent输入的state进行规范化
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)
        # 计算每个注意力头的维度，通过将隐藏层的维度除以注意力头数得到
        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList() # 存储每个注意力头的键（key）提取器
        self.selector_extractors = nn.ModuleList()# 存储每个注意力头的选择器（selector）提取器
        self.value_extractors = nn.ModuleList()# 存储每个注意力头的值（value）提取器
        for i in range(attend_heads):
            '''
            偏置项可以提供额外的灵活性和模型拟合能力。
            但也有一些情况下，不需要偏置项可以简化模型并减少参数量
            '''
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,attend_dim),nn.LeakyReLU()))
        '''
        包含键提取器列表、选择器提取器列表、值提取器列表和评论者编码器列表。
        这些模块是共享的，因为它们在整个模型的训练中会被多次使用
        '''
        self.shared_modules = [self.key_extractors, self.selector_extractors,
                            self.value_extractors, self.critic_encoders]
    '''
    返回模型中共享的参数
    返回一个迭代器，迭代器中包含了所有共享模块（键提取器、选择器提取器、值提取器和评论者编码器）的参数
    '''
    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """ # 使用了Python的列表推导式和chain函数
        # m.parameters()返回一个模块m中的所有参数。使用*操作符将这个列表解包，得到参数的序列
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self): # 缩放共享参数的梯度
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        在多智能体强化学习中，由于每个智能体共享部分模型参数，这些共享参数在每个智能体的批量更新中都会累积梯度。
        为了避免梯度累积导致共享参数的梯度变得过大，需要对这些共享参数的梯度进行缩放
        """
        for p in self.shared_parameters(): # 获取所有共享模块的参数
            '''
            将共享参数的梯度数据（p.grad.data）乘以一个缩放因子，该因子是1除以智能体数量（self.nagents）
            将共享参数的梯度均匀地分摊给每个智能体，避免了梯度累积导致的过度放大问题
            '''
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): 每个代理编码器的输入(batch of obs + ac)
                包含每个智能体的观察值和动作值的 PyTorch 矩阵列表。
                每个元素是一个元组，包含智能体的观察值和动作值
            agents (int): 要返回 Q 的代理索引 (如果为None，表示为所有智能体计算 Q 值)
            return_q (bool): 是否返回 Q 值
            return_all_q (bool): 是否返回所有agent的 Q 值
            regularize (bool): 是否返回正则化项的值
            return_attend (bool): 是否返回每个agent的注意力权重
            logger (TensorboardX SummaryWriter): 如果传入，会将重要值记录到TensorboardX中
            niter（整数）：用于记录到 TensorboardX 的迭代次数
        """
        # 输入的状态（states）和动作（actions）数据分别编码
        # 准备用于多头注意力计算的键（keys）、选择器（selectors）和数值（values）
        if agents is None: #未指定要为其计算 Q 值的智能体索引，那么默认为所有智能体
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps] # 从输入的inps中提取所有智能体的状态，存储在states列表中
        actions = [a for s, a in inps]
        # 将每个智能体的状态和动作连接（concatenate）在一起，形成输入数据
        # 每个智能体的状态和动作成为一个整体，便于后续的编码和计算。
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # 得到每个智能体的状态-动作编码值（sa_encodings列表）
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # 得到每个智能体的状态编码值（s_encodings列表）。这里使用了之前存储的agents变量，表示只对指定的智能体计算 Q 值
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        '''
        将每个智能体的编码值转换为键，存储在all_head_keys列表中
        每个元素是一个列表，包含每个智能体在不同头上的键
        '''
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # 将每个智能体的编码值转换为值，存储在all_head_values列表中,每个元素是一个列表，包含每个智能体在不同头上的值
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # 每个注意力头，将每个智能体的状态编码值转换为选择器,每个元素是一个列表，包含每个智能体在不同头上的选择器
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                            for sel_ext in self.selector_extractors]
        # 三个空列表的列表,用于在迭代过程中动态地保存多个代理（agents）的值
        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head 注意力机制的计算
        for curr_head_keys, curr_head_values, curr_head_selectors in zip( #迭代每个头的键、值和选择器
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            '''
            i 是代理的索引，a_i 是代理的选择器，selector 是当前注意力头的选择器
            '''
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                #通过列表解析创建了不包含当前代理的键和值的列表
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents 计算注意力机制的logits,使用矩阵乘法计算注意力机制的logits
                '''
                注意力机制中的加权内积操作
                使用 view 函数将选择器 selector 的形状进行变换,selector.shape[0] 表示选择器的数量
                1 表示在第二个维度上添加一个维度（这个维度将用于与关键的加权内积）
                -1 表示剩余的维度将被自动计算以保持张量的总大小不变
                '''
                #permute(1, 2, 0) 将维度重新排列为 (key维度, 选择器数量, 代理数量)，这样就可以与选择器的形状匹配
                #attend_logits 是一个张量，它包含了每个选择器与每个关键之间的加权内积值
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                                torch.stack(keys).permute(1, 2, 0))# 将键(keys)按照正确的形状进行排列
                # scale dot-products by size of key (from Attention is All You Need)
                # 对logits进行缩放并计算注意力权重,keys[0].shape[1]表示关键的维度大小
                '''
                内积的结果可以非常大，特别是当关键的维度较大时，这可能导致softmax函数的输入值非常大，进而影响到模型的稳定性
                '''
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2) # 每个选择器对所有关键的注意力权重
                '''
                values中包含了其他代理的对应关键的值
                得到了每个选择器对于其他代理关键的加权和
                将所有关键的加权和求和，得到了每个选择器对于其他代理的加权值other_values
                '''
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                '''
                计算得到的每个选择器对于其他代理的加权值other_values添加到other_all_values列表中.
                该列表用于保存所有选择器对于其他代理的加权值
                将经过缩放的加权内积值attend_logits和softmax得到的注意力权重attend_weights
                保存到all_attend_logits和all_attend_probs列表中，这些列表用于保存每个选择器的注意力权重
                '''
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        '''
        计算了每个代理的Q值（或者其他返回值，取决于参数return_q和return_all_q的设置）
        在需要的情况下，还会计算并返回注意力权重以及注意力熵
        Critic部分
        '''
        all_rets = [] # 存储每个代理的返回值
        for i, a_i in enumerate(agents): #迭代该代理的所有选择器的注意力权重
            # 计算了每个选择器（即注意力头）的注意力熵
            '''
            squeeze()函数在第一个轴用于去除张量中维度大小为1的维度
            sum(1)的操作将每个样本的所有选择器的注意力权重相加，得到了一个标量值，代表了该样本的总注意力分配情
            '''
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                            .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            #该代理的编码输入与其他代理的加权值连接起来，作为Critic网络的输入
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in) # critic_in传入该代理的Critic网络，得到所有动作的Q值
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]# 找出该代理选择的动作中Q值最大的动作的索引
            q = all_q.gather(1, int_acs) # 根据动作索引，从所有动作的Q值中挑选出该代理选择的动作的Q值
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q) #如果需要返回所有动作的Q值,将该代理的所有Q值添加到`agent_rets`列表中
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i]) #所有选择器的注意力内积的平方和作为正则化项。
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend: #是否需要返回注意力权重
                agent_rets.append(np.array(all_attend_probs[i])) #每个选择器的注意力权重转换为NumPy数组，并添加到agent_rets列表中
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                niter) #如果提供了logger对象，将每个选择器的注意力熵记录到日志中
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0]) #如果只有一个返回值，将该值添加到`all_rets`中
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
