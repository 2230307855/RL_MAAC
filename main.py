# 用于训练多智能体强化学习模型的主要训练脚本
import argparse # argparse用于解析命令行参数
import torch
import os
import numpy as np # 处理数值计算，特别适用于多维数组和矩阵运算
from gym.spaces import Box, Discrete # 定义不同类型空间（例如连续空间和离散空间）的类，适用于强化学习环境
from pathlib import Path # 操作文件路径的类，提供了更直观和方便的路径处理方法
from torch.autograd import Variable # 包装张量，支持自动求导c
from tensorboardX import SummaryWriter # 将训练过程中的数据写入TensorBoard日志文件，便于实时可视化和分析训练过程
from utils.make_env import make_env # 创建环境
from utils.buffer import ReplayBuffer # 经验回放缓冲区
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv # 含了对环境进行包装以实现并行化处理的功能
from algorithms.attention_sac import AttentionSAC # 包含了多智能体强化学习算法AttentionSAC的实现

# 创建多个并行环境实例,在多智能体强化学习中同时运行多个环境
def make_parallel_env(env_id, n_rollout_threads, seed):
    '''
    env_id（环境名称）
    n_rollout_threads（并行运行的环境数量）
    和seed（随机种子）
    '''
    def get_env_fn(rank): # 参数rank，表示环境的排名
        def init_env(): # 初始化每个环境实例
            env = make_env(env_id, discrete_action=True) # 创建具体的环境实例
            # 设置环境的随机种子，以确保在相同的初始状态下，每个环境实例产生的随机数序列是相同的
            # 为了使得在不同的训练轮次或不同的并行线程中，环境的初始状态和随机性保持一致，以便于进行可重复性的实验
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    # 根据n_rollout_threads参数
    # 函数选择性地创建一个DummyVecEnv（用于单线程环境）或者一个SubprocVecEnv（用于多线程环境）
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)]) #!!!环境
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)]) #!!!环境

# 整个训练过程的逻辑，包括环境初始化、模型初始化、训练循环、日志记录以及模型保存
def run(config):
    # Path('./models')创建了一个Path对象，该对象指向当前工作目录下的models文件夹
    # 使用/运算符将目录路径与其他部分连接起来
    model_dir = Path('./models') / config.env_id / config.model_name
    # 确定当前的运行编号（run_num），以便将新的训练结果保存到不同的目录
    # 检查之前保存的训练结果目录中的最大运行编号，并在其基础上加1
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                                    model_dir.iterdir() if
                                    str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1 # 如果不存在已保存的运行编号，则将运行编号设置为1
        else:
            run_num = max(exst_run_nums) + 1 # 选择最大的已保存运行编号并加1
    # 根据当前的运行编号构建当前运行的目录路径run_dir
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    # 创建日志目录路径log_dir
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir) # 创建日志目录
    logger = SummaryWriter(str(log_dir)) # 创建SummaryWriter对象，用于记录训练过程中的日志信息

    # 设置随机种子
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    # 创建并初始化环境对象
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    # 使用环境对象创建模型对象!!!(AttentionSAC)
    model = AttentionSAC.init_from_env(env,
                                    tau=config.tau,
                                    pi_lr=config.pi_lr,
                                    q_lr=config.q_lr,
                                    gamma=config.gamma,
                                    pol_hidden_dim=config.pol_hidden_dim,
                                    critic_hidden_dim=config.critic_hidden_dim,
                                    attend_heads=config.attend_heads, #注意力头的数量
                                    reward_scale=config.reward_scale) #奖励范围
    # 创建回放缓冲区对象!!!(replay_buffer)，存储训练过程中的经验数据
    '''
    * config.buffer_length: 缓冲区的最大长度，即可以存储的经验数量
    * model.nagents: 模型中代理（agent）的数量
    * [obsp.shape[0] for obsp in env.observation_space]: 观测空间中每个观测的维度列表
                env.observation_space 表示环境的观测空间.
                obsp.shape[0] 表示每个观测的第一个维度的大小，即观测的维度
    * acsp.shape[0] 表示动作的维度，如果动作空间是连续的（Box 类型），则使用 acsp.shape[0]
                如果动作空间是离散的（Discrete 类型），则使用 acsp.n 表示离散动作的数量    
    '''
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                [obsp.shape[0] for obsp in env.observation_space],
                                [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                for acsp in env.action_space])
    t = 0 # 用于跟踪总共经历的时间步数
    '''
    迭代训练多个轮次（episodes）,config.n_episodes 表示总的训练轮次数
    .n_rollout_threads 表示每个轮次中并行执行的环境数
    '''
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        # 打印当前进行的轮次范围
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset() # 重置环境并获取初始观测
        # 准备模型进行轮次的训练，设置模型的状态为训练模式，并将模型参数放置在CPU上
        model.prep_rollouts(device='cpu')
        # 表示每个轮次的最大时间步数
        for et_i in range(config.episode_length):
            # 将观测数据按照代理（agent）进行重新排列，并转换为 torch 张量
            # np.vstack() 函数将多个观测堆叠成垂直方向的张量
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                requires_grad=False)
                        for i in range(model.nagents)]
            # 使用模型根据观测获取动作，explore=True 表示进行探索
            torch_agent_actions = model.step(torch_obs, explore=True)
            # 将模型输出的动作转换为 NumPy 数组
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # 重新排列动作数据，按照环境进行分组
            # 每个元素 ac 是一个包含所有智能体动作的列表（agent_actions 的元素是智能体个数，每个智能体选择一个动作）
            # actions 是一个二维列表，其中每一行表示一个时间步，每列表示一个环境实例（智能体）在该时间步选择的动作
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions) # 执行动作并获取下一个观测、奖励、完成标志和其他信息
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones) # 经验数据存储到回放缓冲区
            obs = next_obs
            t += config.n_rollout_threads # 更新时间步数计数器，多个线程并行相当于走了n_rollout_threads步
            # 检查是否满足更新模型的条件，即回放缓冲区中的经验数量达到批量大小，并且满足更新频率
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                # 根据配置设置，准备模型进行训练，将模型参数放置在 GPU 或 CPU 上
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates): # 迭代进行模型的更新
                    # ：从回放缓冲区中采样一批经验数据用于模型的更新，config.batch_size 表示每次采样的批量大小
                    sample = replay_buffer.sample(config.batch_size,
                                                to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger) # 更新模型的评论家（critic）部分，用于估计值函数
                    model.update_policies(sample, logger=logger) # 更新模型的策略部分，用于生成动作策略
                    model.update_all_targets() # 更新模型的所有目标网络，用于稳定训练
                model.prep_rollouts(device='cpu') # 准备模型进行轮次的推断（inference），设置模型的状态为推断模式
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads) # 计算当前轮次中每个代理的平均回报
        for a_i, a_ep_rew in enumerate(ep_rews): # 遍历每个代理的平均回报
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                            a_ep_rew * config.episode_length, ep_i)
        # 检查是否达到保存模型的条件，即当前轮次的索引除以保存间隔是否小于并行执行的环境数
        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu') # 准备模型进行轮次的推断（inference），设置模型的状态为推断模式
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    # 使用argparse库来解析命令行参数，并将其存储在config对象中。
    # 然后，使用config对象作为参数来调用run函数。
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                            "model/training contents")
    # 每个训练步骤中并行执行的模拟环境的数量
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    # 添加一个可选参数buffer_length，默认值为1百万，表示经验回放缓冲区的最大容量
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    # 要进行的总训练轮次数
    parser.add_argument("--n_episodes", default=50000, type=int)
    # 每个训练轮次中的最大时间步数
    parser.add_argument("--episode_length", default=25, type=int)
    # 添加一个可选参数steps_per_update，默认值为100，表示每个训练更新步骤中的时间步数
    parser.add_argument("--steps_per_update", default=100, type=int)
    # 每个更新周期中的训练更新次数
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int) # 保存模型和训练内容的间隔步数
    parser.add_argument("--pol_hidden_dim", default=128, type=int) # 策略网络的隐藏层维度
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int) # 注意力机制中的注意头数
    parser.add_argument("--pi_lr", default=0.001, type=float) # 表示策略网络的学习率
    parser.add_argument("--q_lr", default=0.001, type=float)
    # 可选参数tau，默认值为0.001，表示软更新中的目标网络权重更新速率
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float) # 表示回报的缩放因子
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args() # 解析命令行参数，并将结果存储在config对象中

    run(config)
