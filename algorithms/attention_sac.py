import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
'''
Soft Actor-Critic（SAC）算法的实现
用于多智能体强化学习，其中包含了一个中央注意力评论家。
'''

MSELoss = torch.nn.MSELoss() #均方误差

class AttentionSAC(object):
    """
    SAC 代理的包装类，在多代理任务中具有中心注意力批评器
    用于创建多智能体环境下的Soft Actor-Critic (SAC) 算法的实例
    """
    def __init__(self, agent_init_params, sa_size,
                gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                reward_scale=10.,
                pol_hidden_dim=128, # 策略网络的隐藏层维度
                critic_hidden_dim=128, # 评论者网络的隐藏层维度
                attend_heads=4,# 注意力头的数量
                 **kwargs): # 其他可选参数，如果需要的话
        """
        Inputs:
            agent_init_params (list of dict): 初始化每个智能体的参数字典的列表
                num_in_pol (int): 策略的输入维度
                num_out_pol (int): 将维度输出到策略
            sa_size (list of (int, int)): 每个智能体的状态空间和动作空间的大小
            gamma (float): 折扣因子
            tau (float): 目标网络软更新的速率
            pi_lr (float): 策略网络的学习率
            q_lr (float): 评论者网络的学习率
            reward_scale (float): 奖励信号的缩放因子，影响最优策略的熵。默认值为 10.0
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)
        #多个agent
        self.agents = [AttentionAgent(lr=pi_lr, # 创建了一个智能体（AttentionAgent）的列表
                                    hidden_dim=pol_hidden_dim,
                                      **params)
                        for params in agent_init_params]
        # 是主 critic 网络，用于评估当前策略的性能和生成 Q 值。
        # 在训练过程中，它的参数会被更新以最小化 Q 值的误差
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                    attend_heads=attend_heads)
        # 目标 critic 网络，用于计算目标 Q 值。它的参数会被定期更新，以便在训练过程中稳定目标的计算
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                            attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic) #后面的参数直接复制到前面
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                    weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # 都设置到cpu
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0 # 用来跟踪训练的迭代次数或者训练的步数，随着训练的进行而逐渐增加，记录训练的进度

    @property # @property装饰器用于将一个方法转换为只读属性
    def policies(self): # 访问self.policies时，实际上会调用policies方法并返回其结果
        return [a.policy for a in self.agents] # 返回所有代理策略

    @property # 返回所有代理的目标策略
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent（每个代理的观察结果列表）
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                            observations)]
    # 更新中心评论家
    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = [] # 存储下一个动作和对应的对数概率
        next_log_pis = [] #遍历目标策略（target policies）和下一个观察值，计算下一个动作和对数概率
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        # 列表，其中的每个元素是一个张量，代表每个智能体使用目标评论家网络预测的下一个状态-动作对的Q值
        next_qs = self.target_critic(trgt_critic_in)
        # critic_rets 是一个列表，其中的每个元素是一个元组，包含当前评论家网络预测的当前状态-动作对的Q值 (pq) 和正则化项 (regs)
        critic_rets = self.critic(critic_in, regularize=True,
                                logger=logger, niter=self.niter)
        q_loss = 0 #更新中央评论家（central critic）的损失函数，并执行反向传播及优化
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                            next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1))) #目标Q值
            if soft:#如果采用软更新策略，Q值目标进一步减去对数概率 log_pi 除以奖励缩放因子，用于鼓励探索，使得策略更加随机
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach()) # 计算Q值的均方误差损失，当前的Q值与目标Q值的均方误差添加到q_loss中
            for reg in regs:
                q_loss += reg  # regularizing attention  将正则化项加入Q值的损失中，用于对注意力机制进行正则化
        q_loss.backward() # 计算Q值损失相对于网络参数的梯度
        self.critic.scale_shared_grads() # 共享的梯度进行缩放
        grad_norm = torch.nn.utils.clip_grad_norm( # 对梯度进行裁剪，以防止梯度爆炸
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step() #优化器更新评论家网络的参数
        self.critic_optimizer.zero_grad()

        if logger is not None: # 如果提供了日志记录器 logger，将Q值损失写入日志
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1 # 更新迭代次数，用于日志记录
    # 更新智能体策略（policy）
    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        samp_acs = [] # 存储每个智能体的采样动作
        all_probs = [] # 存储每个智能体的策略概率
        all_log_pis = [] # 存储每个智能体的动作对数概率
        all_pol_regs = [] # 存储每个智能体的策略正则化项

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi( # 调用当前智能体的策略，得到
                # 当前状态下动作，策略概率，动作对数概率策略正则化项和策略的熵
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent, # 智能体的策略熵（entropy）写入日志，用于后续分析
                            self.niter)
            samp_acs.append(curr_ac) # 当前状态下智能体的动作加入samp_acs列表
            all_probs.append(probs)
            all_log_pis.append(log_pi) # 动作对数概率加入all_log_pis列表
            all_pol_regs.append(pol_regs)#  将策略正则化项加入all_pol_regs列表

        critic_in = list(zip(obs, samp_acs)) #  构建评论家网络的输入，将状态和采样动作配对
        critic_rets = self.critic(critic_in, return_all_q=True) # 计算所有智能体的当前状态下的Q值
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            '''
            q: 是当前智能体的Q值,是当前评论家网络（Critic）估计的值函数，用于指导策略网络的更新，以便优化预期回报
            all_q: 是在当前状态下，所有智能体的 Q 值组成的张量,包含了每个智能体在当前状态下执行所有可能动作后的 Q 值
                捕捉了所有智能体的动作对环境的影响, 用于计算策略梯度,与策略概率（probs）相乘，并求和，用来估计当前状态
                下的动作价值（value）。这个值被用于计算策略优化的目标，帮助智能体学习更好的策略
            '''
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True) # 计算当前状态下的价值（value），通过将Q值和策略概率相乘后求和得到
            pol_target = q - v #  计算策略目标，用Q值减去价值
            if soft: # 鼓励策略网络学习到在给定状态下选择动作的概率分布，以便最大化预期回报
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else: # 最大化在给定状态下选择动作的概率
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss 不希望批评者从策略损失中积累梯度
            disable_gradients(self.critic) # 确保策略网络的梯度不会影响到批评者（critic）的参数更新
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm( # 梯度裁剪,防止梯度爆炸，阈值为0.5
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None: #当前代理（agent_i）的策略损失（pol_loss）和策略网络梯度的范数（grad_norm）记录到日志中
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                grad_norm, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents: #更新每个网络的策略
            soft_update(a.target_policy, a.policy, self.tau)

    '''
    # 准备模型进行训练，深度学习中，通常需要将模型的状态设置为训练模式 (train())，以便模型能够参与梯度计算和参数更新
    #模型的状态可以通过 train() 和 eval() 方法进行切换
    '''
    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
    # 准备模型进行rollouts（模型在环境中的实际运行）,模型处于评估模式（模型更稳定——不会进行梯度计算和参数更新）
    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval() # 所有智能体的策略网络，设置为评估模式
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename): # 保存训练好的智能体模型的参数
        """
        Save trained parameters of all agents into one file
        """ # 训练过程中，可能使用了GPU加速，移到CPU上，以便于在其他地方加载模型时不依赖特定的GPU设备
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                    'agent_params': [a.get_params() for a in self.agents],
                    'critic_params': {'critic': self.critic.state_dict(),
                                    'target_critic': self.target_critic.state_dict(),
                                    'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename) # 将对象保存到文件的函数。save_dict：要保存的对象或对象字典，filename：保存文件的路径和文件名
    '''
    @classmethod:表示函数是一个类方法，而不是一个实例方法，可以在不创建类实例的情况下被类本身调用
    cls 参数允许你在方法内部使用类的属性和方法
    实例方法的第一个参数是实例本身（通常命名为 self），而类方法的第一个参数是类本身（通常命名为 cls）
    '''
    @classmethod #根据给定的多智能体环境（Multi-agent Gym environment）初始化一个该类的实例
    def init_from_env(cls, env, gamma=0.95, tau=0.01, # cls 参数允许你在方法内部使用类的属性和方法
                    pi_lr=0.01, q_lr=0.01,
                    reward_scale=10.,
                    pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment
        env: Multi-agent Gym environment #环境是多智能体环境
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = [] #存储每个智能体的初始化参数和状态-动作空间大小
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                            env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],#对应观察空间的维度（obsp.shape[0]表示观察空间的第一个维度的大小
                                    'num_out_pol': acsp.n})# 对应动作空间的维度（acsp.n表示动作空间的离散动作的数量)
            sa_size.append((obsp.shape[0], acsp.n)) # sa_size是一个列表，每个ele是元组，这个元组表示每个智能体的观察空间和动作空间的大小

        init_dict = {'gamma': gamma, 'tau': tau, # 所有需要用于初始化类实例的参数和信息
                    'pi_lr': pi_lr, 'q_lr': q_lr,
                    'reward_scale': reward_scale,
                    'pol_hidden_dim': pol_hidden_dim,
                    'critic_hidden_dim': critic_hidden_dim,
                    'attend_heads': attend_heads,
                    'agent_init_params': agent_init_params,
                    'sa_size': sa_size}
        # **操作符，将字典中的键值对作为关键字参数传递给类的构造函数，以便于初始化实例
        instance = cls(**init_dict)
        instance.init_dict = init_dict # 初始化参数保存在实例的init_dict属性中，以便后续参考
        return instance

    @classmethod # 通过save方法保存的文件中实例化该类的对象
    def init_from_save(cls, filename, load_critic=False):
        """
        filename：要加载的文件的路径
        load_critic：是否加载批评者（critic）的参数
        """
        save_dict = torch.load(filename) # 从指定的文件中加载保存的字典
        # **save_dict['init_dict']表示将init_dict字典中的键值对作为关键字参数传递给类的构造函数
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic: # 如果load_critic为True，则继续加载批评者（critic）的参数
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance