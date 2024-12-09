import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.optim import Adam
import random
from gymnasium.envs.registration import register
import time
import  core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
import math
#from envs.HalhCheetah_pret import HalfCheetahEnv
torch.autograd.set_detect_anomaly(True)
import wandb

def evaluate(eval_env, env_steps_count,ac, performance_actor_new):
    evalReturn = 0
    evalIters=1
    borders = []
    colors=[]
    for i in range(evalIters):
        d = False
        steps = 0
        o, _ = eval_env.reset()
        while(not (d or (steps%500==0 and steps != 0)) ):
            a, a_h, b_h, v = ac.stepEval(torch.as_tensor(o, dtype=torch.float32))
            if performance_actor_new is not None:
                actions_per, _, _ = performance_actor_new.actor.get_action(torch.Tensor(o).to("cuda:0").unsqueeze(0))
                actions_per = actions_per.detach().cpu().numpy()
                a,filtered,projected = ac.filter_actions_from_numpyarray(a_h,b_h,actions_per[0])
            next_o, r, d,truncated, info = eval_env.step(a)
            evalReturn+=r
            steps +=1
            #eval_env.render()
            if(a_h>0):
                to_right_is_dangerous = True
            else:
                to_right_is_dangerous = False
            threshold = (b_h/a_h)[0]
            borders.append([o[0],o[2],to_right_is_dangerous, threshold])
            colors.append(steps/500)
            o = next_o

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    arrowDirX=[]
    arrowDirY=[]
    for _,_,to_right_is_dangerous, threshold in borders:
        intensity = min(abs(threshold),1)
        if to_right_is_dangerous:
            if threshold<0:
                #Strong decision drive left(negative) <------
                arrowDirX.append(-1+ intensity * -20)
                arrowDirY.append(0)
            else:
                arrowDirX.append(0)
                arrowDirY.append(0.1)
        else:
            if threshold<0:
                arrowDirX.append(0)
                arrowDirY.append(-0.1)
            else:
                #Strong decision drive right(positive) <------
                arrowDirX.append(1+intensity*20)
                arrowDirY.append(0)
    borders = np.array(borders)
    safe_x = 2.4
    safe_radians = 24 * 2 * math.pi / 360
    rectangle = patches.Rectangle((-safe_x, -safe_radians), 2*safe_x, 2*safe_radians, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    quiver_plot=plt.quiver(borders[:,0], borders[:,1], arrowDirX, arrowDirY,colors, cmap="viridis", angles='xy', scale_units='xy', scale=25)
    # This are the borders of the simulation
    plt.axis([-2*safe_x, 2*safe_x, -2*safe_radians, 2*safe_radians])
    plt.colorbar(quiver_plot, label="Timesteps")            
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Cartpole Border Decisions")
    evalReturn/=evalIters
    # Log the plot to WandB
    wandb.log(data={"agent_eval_safety/env_step": env_steps_count,
                    "agent_eval_safety/episode_reward": evalReturn,
                    "agent_eval_safety/CartpoleBorderDecisions": wandb.Image(plt)},
            step=env_steps_count)
    plt.close()  # Close plot to avoid replotting issues
def evaluate2(eval_env, env_steps_count, ac, performance_actor_new):
    evalReturn = 0
    evalReturnWithBonus = 0
    evalSteps = 500
    max_reward = 1
    evalIters=5
    borders = []
    colors=[]
    filtered = 0
    clipped = 0
    for i in range(evalIters):
        d = False
        steps = 0
        o, _ = eval_env.reset()
        while(not (d or (steps%evalSteps==0 and steps != 0)) ):
            a, a_h, b_h, v = ac.stepEval(torch.as_tensor(o, dtype=torch.float32))
            if performance_actor_new is not None:
                actions_per, _, _ = performance_actor_new.actor.get_action(torch.Tensor(o).to("cuda:0").unsqueeze(0))
                actions_per = actions_per.detach().cpu().numpy()
                a,f,c = ac.filter_actions_from_numpyarray(a_h,b_h,actions_per[0])
                if f:
                    filtered = filtered +1
                if c:
                    clipped = clipped+1
            next_o, r, d,truncated, info = eval_env.step(a)
            evalReturnWithBonus+=r
            evalReturn+=r- info["bonus"]
            steps +=1
            #eval_env.render()
            if(a_h>0):
                to_right_is_dangerous = True
            else:
                to_right_is_dangerous = False
            threshold = (b_h/a_h)[0]
            borders.append([o[0],o[2],to_right_is_dangerous, threshold])
            colors.append(steps/500)
            o = next_o
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    arrowDirX=[]
    arrowDirY=[]
    for _,_,to_right_is_dangerous, threshold in borders:
        intensity = min(abs(threshold),1)
        if to_right_is_dangerous:
            if threshold<0:
                #Strong decision drive left(negative) <------
                arrowDirX.append(-1+ intensity * -20)
                arrowDirY.append(0)
            else:
                arrowDirX.append(0)
                arrowDirY.append(0.1)
        else:
            if threshold<0:
                arrowDirX.append(0)
                arrowDirY.append(-0.1)
            else:
                #Strong decision drive right(positive) <------
                arrowDirX.append(1+intensity*20)
                arrowDirY.append(0)
    borders = np.array(borders)
    safe_x = 2.4
    safe_radians = 24 * 2 * math.pi / 360
    rectangle = patches.Rectangle((-safe_x, -safe_radians), 2*safe_x, 2*safe_radians, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    quiver_plot=plt.quiver(borders[:,0], borders[:,1], arrowDirX, arrowDirY,colors, cmap="viridis", angles='xy', scale_units='xy', scale=25)
    # This are the borders of the simulation
    plt.axis([-2*safe_x, 2*safe_x, -2*safe_radians, 2*safe_radians])
    plt.colorbar(quiver_plot, label="Timesteps")            
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Cartpole Trajectory")
    plot1 = wandb.Image(plt)
    plt.close()

    safe_radians = 24 * 2 * math.pi / 360
    borders = []
    arrowDirX=[]
    arrowDirY=[]
    safe_x=2.4
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colors = []
    colors_v = []
    for x in np.arange(-2.4-2.4,2.5+2.4,0.2):
        for theta in np.arange(-safe_radians*2,safe_radians*2,math.pi / 360 *8):
            o=[x,0,theta,0]
            a, a_h, b_h, v= ac.stepEval(torch.as_tensor(o, dtype=torch.float32))
            colors_v.append(v)
            borders.append([x,theta])
            threshold = np.clip((b_h/a_h)[0],-1,1)
            if(a_h>0):
                to_right_is_dangerous = False
            else:
                to_right_is_dangerous = True
            if to_right_is_dangerous:
                arrowDirX.append(0)
                arrowDirY.append(-1)
                colors.append(threshold)
            else:
                arrowDirX.append(0)
                arrowDirY.append(1)
                colors.append(threshold)
    borders = np.array(borders)
    rectangle = patches.Rectangle((-safe_x, -safe_radians), 2*safe_x, 2*safe_radians, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    quiver_plot=plt.quiver(borders[:,0], borders[:,1], arrowDirX, arrowDirY,colors, cmap ="viridis", angles='xy', scale_units='xy', scale=25)
    # This are the borders of the simulation
    plt.axis([-2*safe_x, 2*safe_x, -2*safe_radians, 2*safe_radians])
    plt.colorbar(quiver_plot, label="Safe actions from value in direction of arrow")            
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Cartpole Border Decisions")
    plot2 = wandb.Image(plt)
    plt.close()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    rectangle = patches.Rectangle((-safe_x, -safe_radians), 2*safe_x, 2*safe_radians, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    value_plot=plt.scatter(borders[:,0], borders[:,1], c= colors_v, cmap ="viridis", s=30)
    # This are the borders of the simulation
    plt.axis([-2*safe_x, 2*safe_x, -2*safe_radians, 2*safe_radians])
    plt.colorbar(value_plot, label="Value function values")            
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Value function")
    plot3 = wandb.Image(plt)
    plt.close()
    evalReturn/=evalIters
    evalReturnWithBonus/=evalIters
    # Log the plot to WandB
    wandb.log(data={"agent_eval_safety/env_step": env_steps_count,
                    "agent_eval_safety/episode_reward": evalReturn,
                    "agent_eval_safety/episode_reward_with_bonus": evalReturnWithBonus,
                    "agent_eval_safety/number_filtered": filtered,
                    "agent_eval_safety/number_clipped": clipped,
                    "agent_eval_safety/CartpoleTrajectory": plot1,
                    "agent_eval_safety/CartpoleBorderDecisions": plot2,
                    "agent_eval_safety/CartpoleValueFunction": plot3},
            step=env_steps_count)
    return evalReturn == evalSteps*max_reward

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.a_h_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.b_h_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        
        self.logp_a_buf = np.zeros(size, dtype=np.float32)
        self.logp_b_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        #buf.store(   o, a, r, c, v,vc, logp)
    def store(self, obs, act, a_h, b_h, rew, val, logp_a, logp_b):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew

        self.val_buf[self.ptr] = val

        self.logp_a_buf[self.ptr] = logp_a
        self.logp_b_buf[self.ptr] = logp_b
        self.a_h_buf[self.ptr] = a_h
        self.b_h_buf[self.ptr] = b_h
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp_a=self.logp_a_buf, logp_b=self.logp_b_buf, a_h=self.a_h_buf, b_h=self.b_h_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs_retrain_threshold=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=1, lagrangian=False, performance_actor_new = None, safe_actor = None, safety_global_step = 0):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    #setup_pytorch_for_mpi()
    import wandb
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    eval_env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    if safe_actor is None:
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    else: 
        ac = safe_actor
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, a_h, b_h, adv, logp_a_old, logp_b_old = data['obs'], data['a_h'], data['b_h'], data['adv'] ,data['logp_a'], data['logp_b']
        # Policy loss
        pi_a, pi_b, logp_a, logp_b = ac.pi(obs, a_h, b_h) # use s-actor for new log prob
        ratio_a = torch.exp(logp_a - logp_a_old)
        ratio_b = torch.exp(logp_b - logp_b_old)

        clip_adv_a = torch.clamp(ratio_a, 1-clip_ratio, 1+clip_ratio) * adv
        clip_adv_b = torch.clamp(ratio_b, 1-clip_ratio, 1+clip_ratio) * adv
        loss_rpi_a = (torch.min(ratio_a * adv, clip_adv_a)).mean()
        loss_rpi_b = (torch.min(ratio_b * adv, clip_adv_b)).mean()
        
        loss_pi = -loss_rpi_a - loss_rpi_b

        # Useful extra info
        approx_kl = (logp_a_old - logp_a).mean().item()
        ent = pi_a.entropy().mean().item()
        clipped = ratio_a.gt(1+clip_ratio) | ratio_a.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()


    # Set up optimizers for policy and value function
    pi_lr = 3e-4
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    
    vf_lr = 1e-3
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

  

    def update(env_steps_count):
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data)
        v_l_old = v_l_old.item()


        # Train policy with multiple steps of gradient descent
        train_pi_iters=80
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        train_v_iters=80
        for i in range(train_v_iters):
            loss_v = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)   # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        wandb.log(data = {
            "agent_train_safety/env_step": env_steps_count,
            "agent_train_safety/lossPi": pi_l_old,
            "agent_train_safety/LossV": v_l_old,
            "agent_train_safety/KL": kl,
            "agent_train_safety/Entropy": ent,
            "agent_train_safety/ClipFrac": cf,
            "agent_train_safety/DeltaLossPi": (loss_pi.item() - pi_l_old),
            "agent_train_safety/DeltaLossV": (loss_v.item() - v_l_old)
        }, step=env_steps_count)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        


    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset()
    ep_ret,ep_cret, ep_len = 0,0,0
    env_steps_count = safety_global_step
    # Main loop: collect experience in env and update/log each epoch
    batch_size= 500
    borders = np.zeros((batch_size,4))
    frames=[]
    epoch = -1
    safety_assured_counter = 0
    while True:
        if safety_assured_counter>=epochs_retrain_threshold:
            wandb.log({"epoch_until_safe": epoch})
            break
        epoch +=1
        for t in range(local_steps_per_epoch):
            a, a_h, b_h, v, logp_a, logp_b = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if performance_actor_new is not None:
                actions_per, _, _ = performance_actor_new.actor.get_action(torch.Tensor(o).to("cuda:0").unsqueeze(0))
                actions_per = actions_per.detach().cpu().numpy()
                a,filtered,projected = ac.filter_actions_from_numpyarray(a_h,b_h,actions_per[0])
            next_o, r, d, truncated, info = env.step(a)
            ep_ret += r
            ep_len += 1
            env_steps_count +=1

            # save and log
            buf.store(o, a, a_h, b_h, r, v, logp_a, logp_b)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, _, _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(last_val=v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, _ = env.reset()
                ep_ret, ep_len = 0,0

        safety_assured = evaluate2(eval_env, env_steps_count, ac, performance_actor_new)
        if safety_assured:
            safety_assured_counter +=1
        else:
            safety_assured_counter = 0
        # Save model
        if (epoch % save_freq == 0) or (safety_assured_counter == epochs_retrain_threshold):
            vals = logger.epoch_dict['EpRet']
            stats = mpi_statistics_scalar(vals, with_min_and_max=True)
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update(env_steps_count)
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    return ac, env_steps_count

def maybe_update_safe_actor(safe_actor_old, performance_actor_new, env_fn, args, safety_global_step, logger_kwargs):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    #if safe_actor_old is None:
    #    num_steps = int(args.s_initial_steps)
    #    epochs = int(num_steps / args.s_steps_per_epoch)
    #else:
    #    epochs = args.s_epoch_retrain
    return ppo(env_fn=env_fn, actor_critic=core.SafeMLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.s_hid]*args.s_l), gamma=args.s_gamma, 
        seed=args.seed, steps_per_epoch=args.s_steps_per_epoch, epochs_retrain_threshold=args.s_epoch_retrain_threshold,
        logger_kwargs=logger_kwargs, lagrangian=False, performance_actor_new = performance_actor_new, safe_actor=safe_actor_old, safety_global_step=safety_global_step)