# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class ActorCriticSAC:
    def __init__(self, envs, device, args):
        self.actor = Actor(envs).to(device)
        self.qf1 = SoftQNetwork(envs).to(device)
        self.qf2 = SoftQNetwork(envs).to(device)
        self.qf1_target = SoftQNetwork(envs).to(device)
        self.qf2_target = SoftQNetwork(envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
        # Automatic entropy tuning
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha
        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.observation_space,
            envs.action_space,
            device,
            handle_timeout_termination=False,
        )

def maybe_update_performance_actor(safe_actor_new, performance_actor_old, env_fn, args, perf_global_step, perf_global_failure_counter, output_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = env_fn()

    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    count_failure = perf_global_failure_counter
    if performance_actor_old is None:
        ac_sac = ActorCriticSAC(envs=envs, device=device, args=args)
    else:
        ac_sac = performance_actor_old

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    return_with_penalty = 0
    obs, _ = envs.reset(seed=args.seed)
    last_start_of_episode = 0
    if performance_actor_old is None:
        end_step = perf_global_step + args.learning_starts
    else:
        end_step = perf_global_step+args.p_retrain_steps
    for global_step in range(perf_global_step, end_step):
        # ALGO LOGIC: put action logic here
        if performance_actor_old is None:
            actions_per = envs.action_space.sample() 
        else:
            actions_per, _, _ = ac_sac.actor.get_action(torch.Tensor(obs).to(device).unsqueeze(0))
            actions_per = actions_per.detach().cpu().numpy()
            actions_per = actions_per[0]
        #FILTER ACTION HERE SAFETY
        _, a_h, b_h, v = safe_actor_new.stepEval(torch.as_tensor(obs, dtype=torch.float32))
        action,_,_ = safe_actor_new.filter_actions_from_numpyarray(a_h,b_h,actions_per)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(action)
        #Penalize reward by deviation from safe action
        if args.penalize_reward:
            rewards = rewards - args.penalize_reward_factor * np.power(np.sum(np.power(actions_per-action,2)),2)
            return_with_penalty += rewards
        if terminations == True:
            count_failure+=1
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if terminations or truncations:
            if "episode" in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                data = {"agent_eval_performance/env_step": global_step,
                    "agent_eval_performance/count_failure":count_failure,
                        "agent_eval_performance/episode_reward": infos['episode']['r'],
                        "agent_eval_performance/episode_reward_with_penalty": return_with_penalty}
                wandb.log(data)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        ac_sac.rb.add(obs, next_obs, actions_per, rewards, terminations, infos)
        if terminations == True:
            terminated_episode = ac_sac.rb._get_samples(range(last_start_of_episode,global_step))
            os.makedirs(os.path.dirname(f"{output_dir}/terminated/"), exist_ok=True)
            with open(f"{output_dir}/terminated/replay_buffer_samples:{last_start_of_episode}_{global_step}.pkl", "wb") as f:
                pickle.dump(terminated_episode, f)
            torch.save(safe_actor_new, f"{output_dir}/terminated/failed_model_{last_start_of_episode}_{global_step}.pt")
            torch.save(ac_sac, f"{output_dir}/terminated/failed_ac_{last_start_of_episode}_{global_step}.pt")
        obs = next_obs
        if terminations or truncations:
            return_with_penalty= 0 
            last_start_of_episode = global_step+1
            obs, infos = envs.reset()
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook

    for global_step_training in range(perf_global_step, end_step):
        # ALGO LOGIC: training  start if performance_actor_o
        if global_step_training>= args.learning_starts:
            data = ac_sac.rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = ac_sac.actor.get_action(data.next_observations)
                qf1_next_target = ac_sac.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = ac_sac.qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - ac_sac.alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = ac_sac.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = ac_sac.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            ac_sac.q_optimizer.zero_grad()
            qf_loss.backward()
            ac_sac.q_optimizer.step()

            if global_step_training % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'policy_frequency' instead of 1
                    pi, log_pi, _ = ac_sac.actor.get_action(data.observations)
                    qf1_pi = ac_sac.qf1(data.observations, pi)
                    qf2_pi = ac_sac.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((ac_sac.alpha * log_pi) - min_qf_pi).mean()

                    ac_sac.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    ac_sac.actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = ac_sac.actor.get_action(data.observations)
                        alpha_loss = (-ac_sac.log_alpha.exp() * (log_pi + ac_sac.target_entropy)).mean()

                        ac_sac.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        ac_sac.a_optimizer.step()
                        ac_sac.alpha = ac_sac.log_alpha.exp().item()

            # update the target networks
            if global_step_training % args.target_network_frequency == 0:
                for param, target_param in zip(ac_sac.qf1.parameters(), ac_sac.qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(ac_sac.qf2.parameters(), ac_sac.qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step_training % 100 == 0:
                data = {
                    "agent_train_performance/env_step": global_step_training,
                    "agent_train_performance/qf1_values": qf1_a_values.mean().item(),
                    "agent_train_performance/qf2_values": qf2_a_values.mean().item(),
                    "agent_train_performance/qf1_loss": qf1_loss.item(),
                    "agent_train_performance/qf2_loss": qf2_loss.item(),
                    "agent_train_performance/qf_loss": qf_loss.item(),
                    "agent_train_performance/actor_loss": actor_loss.item(),
                    "agent_train_performance/alpha": ac_sac.alpha,
                    "agent_train_performance/alpha_loss": alpha_loss.item() if args.autotune else 0
                }
                wandb.log(data)
    #Eval value function of critic
    
    safe_radians = 24 * 2 * math.pi / 360
    borders = []
    safe_x=2.4
    fig = plt.figure(num =1, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(111)
    colors_v = []
    for x in np.arange(-2.4-2.4,2.5+2.4,0.2):
        for theta in np.arange(-safe_radians*2,safe_radians*2,math.pi / 360 *8):
            o=torch.as_tensor([[x,0,theta,0]], dtype=torch.float32, device="cuda:0")
            actions, _, _ = ac_sac.actor.get_action(o)
            qf1_a_values = ac_sac.qf1_target(o, actions).view(-1)
            qf2_a_values = ac_sac.qf2_target(o, actions).view(-1)
            v = (torch.min(qf1_a_values, qf2_a_values)).detach().cpu().numpy()
            colors_v.append(v)
            borders.append([x,theta])
    borders = np.array(borders)
    rectangle = patches.Rectangle((-safe_x, -safe_radians), 2*safe_x, 2*safe_radians, linewidth=2, edgecolor='green', facecolor='white')
    ax.add_patch(rectangle)
    value_plot=plt.scatter(borders[:,0], borders[:,1], c= colors_v, cmap ="viridis", s=30)
    plt.axis([-2*safe_x, 2*safe_x, -2*safe_radians, 2*safe_radians])
    plt.colorbar(value_plot, label="Value function value")            
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.title("Value function performance agent")
    plot1 = wandb.Image(plt)
    wandb.log(data={"agent_train_performance/safeValueFunction": plot1})
    envs.close()
    return ac_sac, global_step, count_failure