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
    exp_name: str = "ppo_humanoid_sparse_opt"
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_run_name: str = "ppo_humanoid_sparse_opt"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "Humanoid-v4"
    total_timesteps: int = 150_000_000

    learning_rate: float = 5e-4
    num_envs: int = 32
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # Speed knobs
    vectorization: str = "async"  # "sync" or "async"
    async_shared_memory: bool = False
    async_context: Optional[str] = None  # e.g., "forkserver" sometimes helps on clusters

    torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default" | "reduce-overhead" | "max-autotune" | "max-autotune-no-cudagraphs"

    set_float32_matmul_precision: str = "high"  # "high" or "highest" if supported
    enable_tf32: bool = True

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class SparseCheckpointHumanoidReward(gym.Wrapper):
    def __init__(
        self,
        env,
        checkpoint_spacing: float = 2.0,
        healthy_z_range: tuple = (1.0, 2.0),
        terminal_bonus_threshold: float = 10.0,
        terminal_bonus: float = 5.0,
        living_reward: float = 0.05,
    ):
        super().__init__(env)
        self.checkpoint_spacing = checkpoint_spacing
        self.healthy_z_range = healthy_z_range
        self.terminal_bonus_threshold = terminal_bonus_threshold
        self.terminal_bonus = terminal_bonus
        self.living_reward = living_reward
        self._highest_checkpoint = 0
        self._initial_x = 0.0

    def _get_torso_height(self, obs):
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
                sparse_reward += float(current_checkpoint - self._highest_checkpoint)
                self._highest_checkpoint = current_checkpoint

        if (terminated or truncated) and (forward_distance >= self.terminal_bonus_threshold):
            sparse_reward += self.terminal_bonus

        if is_healthy:
            sparse_reward += self.living_reward

        info["checkpoints_reached"] = self._highest_checkpoint
        info["forward_distance"] = forward_distance
        return obs, sparse_reward, terminated, truncated, info


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = SparseCheckpointHumanoidReward(
            env,
            checkpoint_spacing=2.0,
            healthy_z_range=(1.0, 2.0),
            terminal_bonus_threshold=10.0,
            terminal_bonus=5.0,
            living_reward=0.05,
        )

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
    def __init__(self, envs):
        super().__init__()
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        act_shape = int(np.prod(envs.single_action_space.shape))
        hidden = 256

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_shape) * -1.0)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        value = self.critic(x)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, value


def maybe_compile(agent: Agent, mode: str):
    agent.actor_mean = torch.compile(agent.actor_mean, mode=mode)
    agent.critic = torch.compile(agent.critic, mode=mode)
    return agent


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.set_float32_matmul_precision)

    if args.enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_fns = [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    if args.vectorization == "async":
        envs = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=args.async_shared_memory,
            context=args.async_context,
        )
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)

    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs).to(device)
    if args.torch_compile and device.type == "cuda":
        agent = maybe_compile(agent, mode=args.compile_mode)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.inference_mode():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.detach().cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

        if isinstance(infos, dict) and "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"].item(), global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"].item(), global_step)
                    writer.add_scalar("charts/checkpoints_reached", info.get("checkpoints_reached", 0), global_step)
                    writer.add_scalar("charts/forward_distance", info.get("forward_distance", 0.0), global_step)
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r'].item():.2f}, "
                        f"len={info['episode']['l'].item()}, checkpoints={info.get('checkpoints_reached', 0)}, "
                        f"distance={info.get('forward_distance', 0.0):.2f}m"
                    )

        with torch.inference_mode():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.inference_mode():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        print("SPS:", sps)

    if args.save_model:
        os.makedirs(f"/scratch/s/shahradm/cleanrl/{run_name}", exist_ok=True)
        model_path = f"/scratch/s/shahradm/cleanrl/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
