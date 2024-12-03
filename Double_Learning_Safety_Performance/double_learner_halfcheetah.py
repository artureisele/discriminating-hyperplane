import os
import random
import time
from dataclasses import dataclass
from gymnasium.envs.registration import register
from envs.cartpole_pret import CartPole
register(
    id="customEnvs/CartPole",
    entry_point="envs.cartpole_pret:CartPole",
)
import gymnasium 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from fppo_double_learnner_halfcheetah import maybe_update_safe_actor
from sac_double_learner_halfcheetah import maybe_update_performance_actor
import warnings
from pathlib import Path
from utils.run_utils import setup_logger_kwargs
from envs.halfcheetah_cost_wrapper import RewardWrapperHalfcheetahHyperPlane
@dataclass
class Args:
    exp_name: str = "DoubleLearningSACHalfcheetahTrainFromFilter"#os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 44
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the environment id of the task"""
    p_retrain_steps: int = 12000
    """total timesteps of the experiments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.05
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    double_learning_iterations = 10
    #Here are arguments for the safe actor learning

    s_hid: int = 256
    """hidden layer size of """
    s_l: int = 2
    """hidden layer number"""
    s_gamma: float = 0.99
    "gamma value for safe actor"
    s_initial_steps: float = 5010000
    "Initial training steps for saftey barriers"
    s_steps_per_epoch: int = 30000
    "Steps in environment per training epoch. If terminated during this steps new episode is started till 4000 is reached"
    s_epoch_retrain: int = 10
    "Number of epochs to retrain safety barriers after every performance actor update"
    safety_filter_default_path = "model_safety_default_halfcheetah.pt"


def make_env_safety_halfcheetah(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gymnasium.make(env_id, render_mode="rgb_array")
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}",step_trigger=lambda x : False)
        else:
            env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = RewardWrapperHalfcheetahHyperPlane(env, safety_reward = True)
        env.action_space.seed(seed)
        return env

    return thunk

def makemake_env_perf(env_id, seed, idx, capture_video, run_name):
    def thunk():
        capture_video = True
        if capture_video and idx == 0:
            env = gymnasium.make(env_id, render_mode="rgb_array")
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x : True)
            print(env.metadata.get("render_fps", None))
        else:
            env = gymnasium.make(env_id)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env = RewardWrapperHalfcheetahHyperPlane(env, safety_reward = False)
        env.action_space.seed(seed)
        return env

    return thunk
def save_safety_actor (safety_actor, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # We are using a non-recommended way of saving PyTorch models,
        # by pickling whole objects (which are dependent on the exact
        # directory structure at the time of saving) as opposed to
        # just saving network weights. This works sufficiently well
        # for the purposes of Spinning Up, but you may want to do 
        # something different for your personal PyTorch project.
        # We use a catch_warnings() context to avoid the warnings about
        # not being able to save the source code.
        torch.save(safety_actor, path)
def log_training_switches(performance:bool):
    if performance:
        wandb.log({"switchToSafety": 0})
        wandb.log({"switchToPerformance": 1})
        print("Switch to Performance")
    else:
        wandb.log({"switchToSafety": 1})
        wandb.log({"switchToPerformance": 0})
        print("Switch to Safety")
if __name__ == "__main__":
    import wandb

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    wandb.define_metric("agent_eval_safety/env_step")
    wandb.define_metric("agent_eval_safety/episode_reward", step_metric="agent_eval_safety/env_step")
    wandb.define_metric("agent_eval_safety/CartpoleBorderDecisions", step_metric="agent_eval_safety/env_step")


    wandb.define_metric("agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/lossPi",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/LossV",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/KL",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/Entropy",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/ClipFrac",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/DeltaLossPi",step_metric="agent_train_safety/env_step")
    wandb.define_metric("agent_train_safety/DeltaLossV", step_metric="agent_train_safety/env_step")


    wandb.define_metric("agent_eval_performance/env_step")
    wandb.define_metric("agent_eval_performance/episode_reward", step_metric="agent_eval_performance/env_step")
    wandb.define_metric("agent_eval_performance/count_failure", step_metric="agent_eval_performance/env_step")


    wandb.define_metric("agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/qf1_values",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/qf2_values",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/qf1_loss",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/qf2_loss",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/qf_loss",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/actor_loss",step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/alpha", step_metric="agent_train_performance/env_step")
    wandb.define_metric("agent_train_performance/alpha_loss", step_metric="agent_train_performance/env_step")

    #Initial Training of safety discriminating hyperplanes
    s_env_fn = make_env_safety_halfcheetah(args.env_id,args.seed,0,args.capture_video,run_name)
    p_env_fn = makemake_env_perf(args.env_id,args.seed,0,args.capture_video,run_name)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.env_id, args.seed)
    output_dir = logger_kwargs["output_dir"]
    log_training_switches(performance=False)

    if not Path(args.safety_filter_default_path).is_file():
        safety_actor, safety_global_step = maybe_update_safe_actor(None, None, s_env_fn, args, 0, logger_kwargs)
        save_safety_actor(safety_actor=safety_actor, path=args.safety_filter_default_path)
    else:
        print(f"Load {args.safety_filter_default_path} safety model")
        safety_actor = torch.load(args.safety_filter_default_path)
        safety_global_step = 0
    log_training_switches(performance=True)

    performance_actor, global_step, count_failure = maybe_update_performance_actor(safety_actor,None, p_env_fn,args,0,0, output_dir)
    print("Initial Learning finished")
    for i in range(0, args.double_learning_iterations):
        log_training_switches(performance=False)
        safety_actor, safety_global_step= maybe_update_safe_actor(safety_actor, performance_actor, s_env_fn, args, safety_global_step,logger_kwargs)
        log_training_switches(performance=True)
        performance_actor, global_step, count_failure = maybe_update_performance_actor(safety_actor, performance_actor, p_env_fn, args,global_step, count_failure, output_dir)