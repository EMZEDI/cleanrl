# dart_dm_control.py
# DART (Dual Adaptive Residual Tracking) for dm_control benchmarks.
# Architecture: 64-64 base critic + 64-64 residual critic, 64-64 actor.
# Uses standard cleanrl wrappers (NormalizeObs/Reward) — no custom sparse wrappers.
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "dm_control/cheetah-run-v0"
    """the id of the environment"""
    total_timesteps: int = 8_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
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
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # DART Specific Arguments
    dart_enabled: bool = True
    """enable DART dual-critic architecture"""
    dart_lambda_res: float = 0.995
    """lambda for the high-bias residual GAE trace"""
    dart_lr_scale: float = 0.3
    """learning rate multiplier for the residual critic"""
    dart_warmup_frac: float = 0.4
    """fraction of total timesteps before residual activates"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, dart_enabled=True):
        super().__init__()
        self.dart_enabled = dart_enabled
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        act_shape = np.prod(envs.single_action_space.shape)

        # Base Critic (V_theta)
        self.critic_base = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Residual Critic (V_phi)
        if self.dart_enabled:
            self.critic_res = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )

        # Actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape))

    def get_value(self, x):
        """V_hat = V_theta + V_phi"""
        v_base = self.critic_base(x)
        if self.dart_enabled:
            v_res = self.critic_res(x)
            return v_base + v_res
        return v_base

    def get_components(self, x):
        """Returns (V_theta, V_phi)"""
        v_base = self.critic_base(x)
        if self.dart_enabled:
            v_res = self.critic_res(x)
        else:
            v_res = torch.zeros_like(v_base)
        return v_base, v_res

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    warmup_steps = int(args.total_timesteps * args.dart_warmup_frac)
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, dart_enabled=args.dart_enabled).to(device)

    # 1. Base Optimizer (Actor + Base Critic)
    base_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd] + list(agent.critic_base.parameters())
    optimizer_base = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)

    # 2. Residual Optimizer (Residual Critic) — scaled LR
    if args.dart_enabled:
        optimizer_res = optim.Adam(
            agent.critic_res.parameters(),
            lr=args.learning_rate * args.dart_lr_scale,
            eps=1e-5,
        )

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Store V_hat (total) and V_theta (base) separately
    values_total = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_base = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_base.param_groups[0]["lr"] = lrnow
            if args.dart_enabled:
                optimizer_res.param_groups[0]["lr"] = lrnow * args.dart_lr_scale

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            with torch.no_grad():
                action, logprob, _, value_hat = agent.get_action_and_value(next_obs)
                v_base_comp, _ = agent.get_components(next_obs)

                # During warmup, V_total = V_base only. After warmup, V_total = V_base + V_res
                active_value_hat = value_hat if is_residual_active else v_base_comp

                values_total[step] = active_value_hat.flatten()
                values_base[step] = v_base_comp.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ===== DART: Dual Return Calculation =====

        # 1. Standard GAE (lambda=0.95) -> Target for Base Critic
        with torch.no_grad():
            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            next_value_hat = (
                agent.get_value(next_obs) if is_residual_active else agent.get_components(next_obs)[0]
            ).reshape(1, -1)

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

            returns_base = advantages + values_total

            # Compute Monte Carlo returns for value accuracy metrics
            mc_returns = torch.zeros_like(rewards).to(device)
            mc_return = 0.0
            for t in reversed(range(args.num_steps)):
                mc_return = rewards[t] + args.gamma * mc_return * (1.0 - dones[t])
                mc_returns[t] = mc_return

            # Value prediction error (MSE between predicted values and MC returns)
            value_pred_error = ((values_total - mc_returns) ** 2).mean().item()
            base_pred_error = ((values_base - mc_returns) ** 2).mean().item()
            mse_improvement = base_pred_error - value_pred_error

            # Unified analysis namespace
            writer.add_scalar("analysis/mse_v_total", value_pred_error, global_step)
            writer.add_scalar("analysis/mse_v_base", base_pred_error, global_step)
            writer.add_scalar("analysis/mse_improvement", mse_improvement, global_step)
            writer.add_scalar("analysis/mean_v_total", values_total.mean().item(), global_step)
            writer.add_scalar("analysis/mean_mc_return", mc_returns.mean().item(), global_step)

            # Backward-compatible aliases
            writer.add_scalar("value_metrics/prediction_mse", value_pred_error, global_step)
            writer.add_scalar("value_metrics/mean_predicted_value", values_total.mean().item(), global_step)
            writer.add_scalar("value_metrics/mean_mc_return", mc_returns.mean().item(), global_step)
            writer.add_scalar("value_metrics/residual_active", float(is_residual_active), global_step)

            # Residual diagnostics (V_res should correct V_base errors)
            values_res = values_total - values_base
            residual_target = mc_returns - values_base
            writer.add_scalar("dart/mean_base_value", values_base.mean().item(), global_step)
            writer.add_scalar("dart/mean_residual_value", values_res.mean().item(), global_step)
            writer.add_scalar("value_metrics/mean_residual_value", values_res.mean().item(), global_step)
            writer.add_scalar(
                "value_metrics/residual_prediction_mse",
                ((values_res - residual_target) ** 2).mean().item(),
                global_step,
            )

            # Advantage-quality diagnostic: Corr(A_hat, A_true)
            b_adv_np = advantages.reshape(-1).detach().cpu().numpy()
            b_true_adv_np = (mc_returns - values_total).reshape(-1).detach().cpu().numpy()
            if np.std(b_adv_np) > 1e-12 and np.std(b_true_adv_np) > 1e-12:
                adv_corr = float(np.corrcoef(b_adv_np, b_true_adv_np)[0, 1])
            else:
                adv_corr = 0.0
            writer.add_scalar("analysis/advantage_correlation", adv_corr, global_step)
            writer.add_scalar("value_metrics/advantage_correlation", adv_corr, global_step)

            # Track base and residual components separately
            if args.dart_enabled:
                writer.add_scalar("value_metrics/mean_base_value", values_base.mean().item(), global_step)
                writer.add_scalar("value_metrics/base_prediction_mse", base_pred_error, global_step)

        # 2. High-Lambda Return (lambda=dart_lambda_res) -> Target for Residual Critic
        if args.dart_enabled:
            with torch.no_grad():
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
                    gae_res = delta_res + args.gamma * args.dart_lambda_res * nextnonterminal * lastgaelam_res
                    lastgaelam_res = gae_res
                    returns_res[t] = gae_res + values_total[t]

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns_base = returns_base.reshape(-1)
        b_values_base = values_base.reshape(-1)  # Frozen Base snapshot

        if args.dart_enabled:
            b_returns_res = returns_res.reshape(-1)

        # Optimization
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        is_residual_active = args.dart_enabled and (global_step > warmup_steps)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                new_v_base, new_v_res = agent.get_components(b_obs[mb_inds])

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

                # Base Critic Loss
                newvalue_base = new_v_base.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue_base - b_returns_base[mb_inds]) ** 2
                    v_clipped = b_values_base[mb_inds] + torch.clamp(
                        newvalue_base - b_values_base[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_base[mb_inds]) ** 2
                    v_loss_base_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss_base = 0.5 * v_loss_base_max.mean()
                else:
                    v_loss_base = 0.5 * ((newvalue_base - b_returns_base[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss_base = pg_loss - args.ent_coef * entropy_loss + v_loss_base * args.vf_coef

                optimizer_base.zero_grad()
                loss_base.backward()
                nn.utils.clip_grad_norm_(base_params, args.max_grad_norm)
                optimizer_base.step()

                # Residual Critic Loss (target = G_high_lambda - V_base_frozen)
                if is_residual_active:
                    newvalue_res = new_v_res.view(-1)
                    target_res = b_returns_res[mb_inds] - b_values_base[mb_inds]

                    loss_res = ((newvalue_res - target_res) ** 2).mean()

                    optimizer_res.zero_grad()
                    loss_res.backward()
                    nn.utils.clip_grad_norm_(agent.critic_res.parameters(), args.max_grad_norm)
                    optimizer_res.step()
                else:
                    loss_res = torch.tensor(0.0)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        y_pred, y_true = (b_values_base + (b_returns_res - b_values_base if args.dart_enabled else 0)).cpu().numpy(), b_returns_base.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer_base.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_base", v_loss_base.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.dart_enabled:
            writer.add_scalar("dart/loss_res", loss_res.item(), global_step)
            writer.add_scalar("dart/is_active", float(is_residual_active), global_step)
            with torch.no_grad():
                mean_res = (b_returns_res - b_values_base).mean().item()
                writer.add_scalar("dart/residual_correction_mean", mean_res, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
