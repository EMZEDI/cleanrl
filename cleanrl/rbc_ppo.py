import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import minigrid
from minigrid.wrappers import FlatObsWrapper
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "rbc_ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation (used for Policy and TD Head)"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.7
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RBC-PPO Specific Arguments
    rbc_enabled: bool = True
    """whether to use Residual Bias-Corrected PPO"""
    rbc_lambda_res: float = 0.99
    """High lambda for the residual head targets (low bias, higher variance)"""
    rbc_lr_scale: float = 0.1
    """Learning rate scale for the residual head (implements the slow time-scale)"""
    rbc_warmup_frac: float = 0.2
    """Fraction of total timesteps to freeze the residual head (Freeze-and-Fine-Tune)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class DictFlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env)
        lows = []
        highs = []
        self._keys = []
        self._subspaces = []
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                self._keys.append(key)
                self._subspaces.append(space)
                lows.append(space.low.astype(np.float32).flatten())
                highs.append(space.high.astype(np.float32).flatten())
            elif isinstance(space, gym.spaces.Discrete):
                self._keys.append(key)
                self._subspaces.append(space)
                lows.append(np.array([0.0], dtype=np.float32))
                highs.append(np.array([float(space.n - 1)], dtype=np.float32))
            else:
                continue
        if not lows:
            raise ValueError("DictFlattenObservation needs at least one numeric subspace")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(lows, axis=0),
            high=np.concatenate(highs, axis=0),
            dtype=np.float32,
        )

    def observation(self, obs):
        parts = []
        for key, space in zip(self._keys, self._subspaces):
            value = obs[key]
            if isinstance(space, gym.spaces.Box):
                parts.append(np.asarray(value, dtype=np.float32).flatten())
            elif isinstance(space, gym.spaces.Discrete):
                parts.append(np.asarray(value, dtype=np.float32).reshape(1))
            else:
                raise NotImplementedError
        return np.concatenate(parts, axis=0)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        if "MiniGrid" in env_id.lower():
            env = FlatObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = DictFlattenObservation(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rbc_enabled=True):
        super().__init__()
        self.rbc_enabled = rbc_enabled
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        
        # Actor Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        # TD Head (Fast Learner)
        self.critic_td = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Residual Head (Slow Learner)
        # Separate network to ensure gradient blocking and clean feature separation
        if self.rbc_enabled:
            self.critic_res = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.5),
            )

    def get_value(self, x):
        """Returns the combined value estimate V_hat = V_theta + R_phi"""
        v_td = self.critic_td(x)
        if self.rbc_enabled:
            v_res = self.critic_res(x)
            return v_td + v_res
        return v_td

    def get_components(self, x):
        """Returns V_theta and R_phi separately"""
        v_td = self.critic_td(x)
        if self.rbc_enabled:
            v_res = self.critic_res(x)
        else:
            v_res = torch.zeros_like(v_td)
        return v_td, v_res

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        # Get combined value for advantage calculation
        value = self.get_value(x)
        
        return action, probs.log_prob(action), probs.entropy(), value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, rbc_enabled=args.rbc_enabled).to(device)
    
    # Separate optimizers or param groups to handle different LRs
    # Standard parameters (Actor + TD Critic)
    base_params = list(agent.actor.parameters()) + list(agent.critic_td.parameters())
    optimizer_base = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)
    
    # Residual parameters (Slow Learner)
    if args.rbc_enabled:
        optimizer_res = optim.Adam(
            agent.critic_res.parameters(), 
            lr=args.learning_rate * args.rbc_lr_scale, 
            eps=1e-5
        )

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Store total value (V_hat) and TD component (V_theta) separately
    values_total = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_td = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_base.param_groups[0]["lr"] = lrnow
            if args.rbc_enabled:
                # Maintain the ratio for residual optimizer
                optimizer_res.param_groups[0]["lr"] = lrnow * args.rbc_lr_scale

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value_hat = agent.get_action_and_value(next_obs)
                v_td_comp, _ = agent.get_components(next_obs)
                
                values_total[step] = value_hat.flatten()
                values_td[step] = v_td_comp.flatten()
                
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ===== RBC-PPO: Dual GAE Computation =====
        
        # 1. Standard GAE for Policy and TD Head (Low Variance, lambda=0.95)
        with torch.no_grad():
            next_value_hat = agent.get_value(next_obs).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value_hat
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values_total[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values_total[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            
            # Returns for TD head target (Standard PPO return)
            returns = advantages + values_total

        # 2. High-Lambda Return for Residual Head (Low Bias, lambda=0.99)
        # Note: The residual head targets (G_high_lambda - V_theta).
        if args.rbc_enabled:
            with torch.no_grad():
                # We calculate a second set of advantages/returns using rbc_lambda_res
                # We still bootstrap from the full estimator V_hat to get the best 1-step estimate
                returns_res = torch.zeros_like(rewards).to(device)
                lastgaelam_res = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value_hat
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values_total[t + 1]
                    
                    delta_res = rewards[t] + args.gamma * nextvalues * nextnonterminal - values_total[t]
                    # GAE calculation with high lambda
                    gae_res = delta_res + args.gamma * args.rbc_lambda_res * nextnonterminal * lastgaelam_res
                    lastgaelam_res = gae_res
                    
                    # G_lambda = A_lambda + V
                    returns_res[t] = gae_res + values_total[t]

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_total.reshape(-1)
        b_values_td = values_td.reshape(-1) # Old V_theta
        
        if args.rbc_enabled:
            b_returns_res = returns_res.reshape(-1)

        # Optimization
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        # Check Freeze Schedule
        is_residual_active = args.rbc_enabled and (global_step > args.total_timesteps * args.rbc_warmup_frac)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue_hat = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                new_v_td, new_v_res = agent.get_components(b_obs[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # TD Head Loss (Fast Learner)
                # Target: Standard returns (b_returns).
                # Note: We use the TD head's own output (new_v_td) for this loss
                newvalue_td = new_v_td.view(-1)
                
                # We use clipped loss logic on the TD head relative to the OLD TD values
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue_td - b_returns[mb_inds]) ** 2
                    v_clipped = b_values_td[mb_inds] + torch.clamp(
                        newvalue_td - b_values_td[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_td_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss_td = 0.5 * v_loss_td_max.mean()
                else:
                    v_loss_td = 0.5 * ((newvalue_td - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                
                # Combined Base Loss
                loss_base = pg_loss - args.ent_coef * entropy_loss + v_loss_td * args.vf_coef

                optimizer_base.zero_grad()
                loss_base.backward()
                nn.utils.clip_grad_norm_(base_params, args.max_grad_norm)
                optimizer_base.step()

                # Residual Head Loss (Slow Learner)
                # Target: High-lambda return MINUS current V_theta.
                # Crucially: V_theta is DETACHED (gradient blocking).
                if is_residual_active:
                    newvalue_res = new_v_res.view(-1)
                    target_res = b_returns_res[mb_inds] - new_v_td.detach().view(-1)
                    
                    # Simple MSE for residual
                    loss_res = ((newvalue_res - target_res) ** 2).mean()
                    
                    optimizer_res.zero_grad()
                    loss_res.backward()
                    nn.utils.clip_grad_norm_(agent.critic_res.parameters(), args.max_grad_norm)
                    optimizer_res.step()
                else:
                    loss_res = torch.tensor(0.0)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer_base.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_td", v_loss_td.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if args.rbc_enabled:
            writer.add_scalar("rbc/loss_res", loss_res.item(), global_step)
            writer.add_scalar("rbc/is_active", float(is_residual_active), global_step)
            writer.add_scalar("rbc/residual_magnitude", b_values_td.abs().mean().item(), global_step)
            # Log bias ratio (Residual / TD)
            mean_res = b_values.mean().item() - b_values_td.mean().item()
            mean_td = b_values_td.mean().item()
            writer.add_scalar("rbc/bias_fraction", mean_res / (abs(mean_td) + 1e-6), global_step)
            # Log correlation between Residual and Return Error
            residual_vals = b_values - b_values_td # effectively R_phi
            td_error = b_returns - b_values_td
            # If R_phi predicts the TD error, this correlation should be positive
            corr = torch.corrcoef(torch.stack((residual_vals, td_error)))[0, 1]
            writer.add_scalar("rbc/residual_correlation", corr, global_step)

    envs.close()
    writer.close()
