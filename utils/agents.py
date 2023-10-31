from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy #离散策略

class AttentionAgent(object):
    """
    注意力代理的常规类(policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): 策略输入的维度数
            num_out_pol (int): 策略输出的维度数
            隐藏层维度 hidden_dim（默认为64）
            one_hot编码的默认维度 0
        """
        #两个策略的初始完全一致
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)
        #将直接参数从源网络转移到目标网络
        hard_update(self.target_policy, self.policy)
        #策略优化器Adam
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        在环境中向前迈进了一步，以进行小批量观察
        Inputs:
            obs (PyTorch Variable): 该代理的观察结果
            explore (boolean): 是否取样
        Outputs:
            action (PyTorch Variable): 此代理的操作
        """
        return self.policy(obs, sample=explore)

    #返回策略网络、目标策略网络和优化器的状态字典
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    #加载给定的参数到代理网络和优化器中
    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
