# DART (Dual Adaptive Residual Tracking) for Sparse Checkpoint Humanoid (Humanoid-v4)
import os
import random
import time
from dataclasses import dataclass

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
    exp_name: str = "dart_humanoid_sparse"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_run_name: str = "dart_humanoid_sparse_v2"
    wandb_entity: str = None
    capture_video: bool = False

    # Save (kept optional; your pendulum script referenced this but didn't define it)
    save_model: bool = False

    # Algorithm specific arguments (UNCHANGED from your pendulum DART script)
    env_id: str = "Humanoid-v4"
    total_timesteps: int = 120000000
    learning_rate: float = 9e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # DART Specific Arguments (UNCHANGED)
    dart_enabled: bool = True
    dart_lambda_res: float = 0.9999
    dart_lr_scale: float = 0.22
    dart_warmup_frac: float = 0.6

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --- SPARSE CHECKPOINT REWARD WRAPPER (Humanoid) ---
class SparseCheckpointHumanoidReward(gym.Wrapper):
    """
    +1 for each new forward checkpoint crossed (spacing meters) while healthy (torso z in range).
    Reward is 0 otherwise.
    Optional terminal bonus if far enough at episode end.
    """
    def __init__(
        self,
        env,
        checkpoint_spacing: float = 0.5,
        healthy_z_range: tuple = (1.0, 2.0),
        terminal_bonus_threshold: float = 10.0,
        terminal_bonus: float = 5.0,
    ):
        super().__init__(env)
        self.checkpoint_spacing = checkpoint_spacing
        self.healthy_z_range = healthy_z_range
        self.terminal_bonus_threshold = terminal_bonus_threshold
        self.terminal_bonus = terminal_bonus

        self._highest_checkpoint = 0
        self._initial_x = 0.0

    def _get_torso_height(self, obs):
        # Robust to observation normalization/clipping wrappers outside this wrapper.
        try:
            return float(self.env.unwrapped.data.qpos[2])
        except Exception:
            return float(obs[0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._highest_checkpoint = 0
        self._initial_x = float(info.get("x_position", 0.0))
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        sparse_reward = 0.0
        x_pos = float(info.get("x_position", 0.0))
        forward_distance = x_pos - self._initial_x

        z_height = self._get_torso_height(obs)
        is_healthy = (self.healthy_z_range[0] <= z_height <= self.healthy_z_range[1])

        if is_healthy and forward_distance > 0:
            current_checkpoint = int(forward_distance / self.checkpoint_spacing)
            if current_checkpoint > self._highest_checkpoint:
                sparse_reward = float(current_checkpoint - self._highest_checkpoint)
                self._highest_checkpoint = current_checkpoint

        if (terminated or truncated) and (forward_distance >= self.terminal_bonus_threshold):
            sparse_reward += self.terminal_bonus

        info["checkpoints_reached"] = self._highest_checkpoint
        info["forward_distance"] = forward_distance

        if is_healthy:
            sparse_reward += 0.001
        return obs, sparse_reward, terminated, truncated, info


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # IMPORTANT: sparse reward wrapper first (before obs normalization)
        env = SparseCheckpointHumanoidReward(
            env,
            checkpoint_spacing=5.0,          # Changed from 2.0 - rewards every 5m instead of 2m
            healthy_z_range=(1.0, 2.0),      # Keep same
            terminal_bonus_threshold=20.0,   # Changed from 10.0 - harder to get bonus
            terminal_bonus=10.0,             # Changed from 5.0 - bigger when you get it
        )

        # Then preprocessing/statistics
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
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
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        act_shape = int(np.prod(envs.single_action_space.shape))

        # Humanoid-sized networks (bigger than pendulum)
        hidden = 256

        # Base Critic (V_theta)
        self.critic_base = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

        # Residual Critic (V_phi)
        if self.dart_enabled:
            self.critic_res = nn.Sequential(
                layer_init(nn.Linear(obs_shape, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, 1), std=1.0),
            )

        # Actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_shape), std=0.01),
        )
        # Slightly smaller initial std to reduce extreme actions under ClipAction
        self.actor_logstd = nn.Parameter(torch.ones(1, act_shape) * -1.0)

    def get_value(self, x):
        v_base = self.critic_base(x)
        if self.dart_enabled:
            v_res = self.critic_res(x)
            return v_base + v_res
        return v_base

    def get_components(self, x):
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
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        value = self.get_value(x)
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), value


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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, dart_enabled=args.dart_enabled).to(device)

    # Base optimizer: Actor + Base critic
    base_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd] + list(agent.critic_base.parameters())
    optimizer_base = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)

    # Residual optimizer: Residual critic only
    if args.dart_enabled:
        optimizer_res = optim.Adam(
            agent.critic_res.parameters(),
            lr=args.learning_rate * args.dart_lr_scale,
            eps=1e-5,
        )

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)

    values_total = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_base = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_base.param_groups[0]["lr"] = lrnow
            if args.dart_enabled:
                optimizer_res.param_groups[0]["lr"] = lrnow * args.dart_lr_scale

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            with torch.no_grad():
                action, logprob, _, value_hat = agent.get_action_and_value(next_obs)
                v_base_comp, _ = agent.get_components(next_obs)
                active_value_hat = value_hat if is_residual_active else v_base_comp

                values_total[step] = active_value_hat.flatten()
                values_base[step] = v_base_comp.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

        # Episode logging (same pattern as CleanRL PPO scripts)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_r = info["episode"]["r"].item()
                    ep_l = info["episode"]["l"].item()
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)
                    writer.add_scalar("charts/checkpoints_reached", info.get("checkpoints_reached", 0), global_step)
                    writer.add_scalar("charts/forward_distance", info.get("forward_distance", 0.0), global_step)
                    print(
                        f"global_step={global_step}, episodic_return={ep_r:.2f}, "
                        f"len={ep_l}, checkpoints={info.get('checkpoints_reached', 0)}, "
                        f"distance={info.get('forward_distance', 0.0):.2f}m"
                    )

        # ===== DART: Dual Return Calculation =====
        with torch.no_grad():
            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            next_value_hat = (
                agent.get_value(next_obs) if is_residual_active else agent.get_components(next_obs)[0]
            ).reshape(1, -1)

            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value_hat
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values_total[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values_total[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            returns_base = advantages + values_total

        if args.dart_enabled:
            with torch.no_grad():
                returns_res = torch.zeros_like(rewards, device=device)
                lastgaelam_res = 0.0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value_hat
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values_total[t + 1]

                    delta_res = rewards[t] + args.gamma * nextvalues * nextnonterminal - values_total[t]
                    lastgaelam_res = delta_res + args.gamma * args.dart_lambda_res * nextnonterminal * lastgaelam_res
                    returns_res[t] = lastgaelam_res + values_total[t]

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns_base = returns_base.reshape(-1)
        b_values_base = values_base.reshape(-1)

        if args.dart_enabled:
            b_returns_res = returns_res.reshape(-1)

        # PPO/DART updates
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
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Base critic loss
                newvalue_base = new_v_base.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue_base - b_returns_base[mb_inds]) ** 2
                    v_clipped = b_values_base[mb_inds] + torch.clamp(
                        newvalue_base - b_values_base[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_base[mb_inds]) ** 2
                    v_loss_base = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss_base = 0.5 * ((newvalue_base - b_returns_base[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss_base = pg_loss - args.ent_coef * entropy_loss + v_loss_base * args.vf_coef

                optimizer_base.zero_grad()
                loss_base.backward()
                nn.utils.clip_grad_norm_(base_params, args.max_grad_norm)
                optimizer_base.step()

                # Residual critic loss (only after warmup)
                if is_residual_active:
                    newvalue_res = new_v_res.view(-1)
                    target_res = b_returns_res[mb_inds] - b_values_base[mb_inds]
                    loss_res = ((newvalue_res - target_res) ** 2).mean()

                    optimizer_res.zero_grad()
                    loss_res.backward()
                    nn.utils.clip_grad_norm_(agent.critic_res.parameters(), args.max_grad_norm)
                    optimizer_res.step()
                else:
                    loss_res = torch.tensor(0.0, device=device)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        y_pred = b_values_base.detach().cpu().numpy()
        y_true = b_returns_base.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer_base.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_base", v_loss_base.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance_base", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.dart_enabled:
            writer.add_scalar("dart/loss_res", loss_res.item(), global_step)
            writer.add_scalar("dart/is_active", float(is_residual_active), global_step)
            if is_residual_active:
                with torch.no_grad():
                    writer.add_scalar(
                        "dart/residual_target_mean",
                        (b_returns_res - b_values_base).mean().item(),
                        global_step,
                    )

    if args.save_model:
        model_path = f"/scratch/s/shahradm/cleanrl/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
