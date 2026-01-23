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
    exp_name: str = "dart_ppo_lstm_flat"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Harder Environment
    env_id: str = "MiniGrid-MultiRoom-N4-S5-v0"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16 
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01 
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # DART Specific
    dart_enabled: bool = True
    dart_lambda_res: float = 0.998
    dart_lr_scale: float = 0.1
    dart_warmup_frac: float = 0.1

    # Runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        # KEY: Using FlatObsWrapper as requested
        env = FlatObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
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
        
        # 1. Feature Extractor
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        # 2. LSTM
        self.lstm = nn.LSTM(128, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 3. Heads
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic_base = layer_init(nn.Linear(128, 1), std=1.0)
        if self.dart_enabled:
            self.critic_res = layer_init(nn.Linear(128, 1), std=1.0)

    def get_states(self, x, lstm_state, done):
        # x: (Total_Steps, Obs_Dim)
        hidden = self.encoder(x)

        # Infer dimensions
        # lstm_state is (1, batch_size, hidden)
        batch_size = lstm_state[0].shape[1]
        seq_len = x.shape[0] // batch_size

        if seq_len == 1:
            # Inference / Rollout mode (Fast)
            # Reshape to (Seq=1, Batch, Feat)
            hidden = hidden.view(1, batch_size, -1)
            
            # Reset state based on done
            h, c = lstm_state
            mask = (1.0 - done).view(1, -1, 1)
            h = h * mask
            c = c * mask
            
            # Forward
            new_hidden, (new_h, new_c) = self.lstm(hidden, (h, c))
            return new_hidden.flatten(0, 1), (new_h, new_c)
        
        else:
            # Training mode (Sequence Learning)
            # Reshape to (Seq, Batch, Feat)
            hidden = hidden.view(seq_len, batch_size, -1)
            done = done.view(seq_len, batch_size)
            
            h, c = lstm_state
            outputs = []
            
            # Manually loop to handle resets inside the batch
            for t in range(seq_len):
                mask = (1.0 - done[t]).view(1, -1, 1)
                h = h * mask
                c = c * mask
                
                step_out, (h, c) = self.lstm(hidden[t].unsqueeze(0), (h, c))
                outputs.append(step_out.squeeze(0))
                
            new_hidden = torch.stack(outputs, dim=0) # (Seq, Batch, Feat)
            return new_hidden.flatten(0, 1), (h, c)

    # Rest of the methods remain similar...
    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        v_base = self.critic_base(hidden)
        if self.dart_enabled:
            v_res = self.critic_res(hidden.detach())
            return v_base + v_res
        return v_base

    def get_components(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        v_base = self.critic_base(hidden)
        if self.dart_enabled:
            v_res = self.critic_res(hidden.detach())
        else:
            v_res = torch.zeros_like(v_base)
        return v_base, v_res

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, next_lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        v_base = self.critic_base(hidden)
        if self.dart_enabled:
            v_res = self.critic_res(hidden.detach())
            value = v_base + v_res
        else:
            value = v_base
            
        return action, probs.log_prob(action), probs.entropy(), value, next_lstm_state


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    agent = Agent(envs, dart_enabled=args.dart_enabled).to(device)
    
    # Optimizer
    # Includes Encoder + LSTM + Actor + Base Critic
    base_params = (
        list(agent.encoder.parameters()) + 
        list(agent.lstm.parameters()) + 
        list(agent.actor.parameters()) + 
        list(agent.critic_base.parameters())
    )
    optimizer_base = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)
    
    if args.dart_enabled:
        optimizer_res = optim.Adam(
            agent.critic_res.parameters(), 
            lr=args.learning_rate * args.dart_lr_scale, 
            eps=1e-5
        )

    # Storage
    # obs_shape is flattened (e.g. 147)
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    values_total = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_base = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Global Step
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # LSTM States (h, c) - initialized to zeros
    # Shape: (1, num_envs, hidden_dim)
    next_lstm_state = (
        torch.zeros(1, args.num_envs, 128).to(device),
        torch.zeros(1, args.num_envs, 128).to(device),
    )

    for iteration in range(1, args.num_iterations + 1):
        # LR Annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_base.param_groups[0]["lr"] = lrnow
            if args.dart_enabled:
                optimizer_res.param_groups[0]["lr"] = lrnow * args.dart_lr_scale

        # Store initial LSTM state for the rollout
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            
            with torch.no_grad():
                action, logprob, _, value_hat, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )
                
                # Separate components logic
                v_base_comp, _ = agent.get_components(next_obs, next_lstm_state, next_done)
                active_value_hat = value_hat if is_residual_active else v_base_comp
                values_total[step] = active_value_hat.flatten()
                values_base[step] = v_base_comp.flatten()
                
            actions[step] = action
            logprobs[step] = logprob

            real_next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(real_next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE Calculation
        with torch.no_grad():
            is_residual_active = args.dart_enabled and (global_step > warmup_steps)
            
            # Get next value for GAE
            next_value_hat = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            if not is_residual_active:
                 next_value_hat = agent.get_components(next_obs, next_lstm_state, next_done)[0].reshape(1, -1)

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

            # DART High Lambda Return
            if args.dart_enabled:
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
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns_base = returns_base.reshape(-1)
        b_values = values_total.reshape(-1)
        b_values_base = values_base.reshape(-1)
        
        if args.dart_enabled:
            b_returns_res = returns_res.reshape(-1)

        # Optimization
        envs_per_batch = args.num_envs // args.num_minibatches
        env_inds = np.arange(args.num_envs) 
        flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        
        clipfracs = []
        is_residual_active = args.dart_enabled and (global_step > warmup_steps)

        for epoch in range(args.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, envs_per_batch):
                end = start + envs_per_batch
                mb_env_inds = env_inds[start:end]
                
                # Get indices for this minibatch
                mb_inds = flat_inds[:, mb_env_inds].flatten()
                
                # Prepare LSTM states for the start of the batch (step 0)
                # Correctly slicing the tuple of tensors (1, num_envs, hidden)
                h_0 = initial_lstm_state[0][:, mb_env_inds, :]
                c_0 = initial_lstm_state[1][:, mb_env_inds, :]
                
                # Forward pass
                _, newlogprob, entropy, newvalue_hat, _ = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    (h_0, c_0), 
                    b_dones[mb_inds], 
                    b_actions.long()[mb_inds]
                )
                
                # Components pass
                new_v_base, new_v_res = agent.get_components(
                    b_obs[mb_inds], 
                    (h_0, c_0), 
                    b_dones[mb_inds]
                )

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

                # Residual Critic Loss (DART)
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns_base.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer_base.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_base", v_loss_base.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if args.dart_enabled:
            writer.add_scalar("dart/loss_res", loss_res.item(), global_step)

    envs.close()
    writer.close()