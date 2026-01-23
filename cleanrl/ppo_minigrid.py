# corrected_cleanrl_minigrid_ppo.py
# Based on your script — fixes for LSTM-done shaping, action dtype, and device-safe tensor creation.

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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # Environment
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

    # Runtime calculated
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
        env = FlatObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        
        # 1. Encoder
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(int(obs_shape), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        # 2. LSTM (Explicit batch_first=False)
        self.lstm = nn.LSTM(128, 128, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 3. Heads
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.encoder(x)

        # Infer dimensions
        batch_size = lstm_state[0].shape[1]
        seq_len = hidden.shape[0] // batch_size

        # Reshape to (seq_len, batch_size, hidden_dim)
        hidden = hidden.view(seq_len, batch_size, -1)
        
        # Prepare done mask (seq_len, batch_size)
        if done is None:
            done = torch.zeros(seq_len, batch_size, device=hidden.device)
        else:
            done = done.to(dtype=torch.float32, device=hidden.device)
            if done.numel() == batch_size:
                done = done.view(1, batch_size) # Rollout case
            else:
                done = done.view(seq_len, batch_size) # Training case

        # --- THE FIX ---
        # We must process step-by-step to handle internal resets
        h, c = lstm_state
        outputs = []
        for t in range(seq_len):
            # If done[t] is True, it means the step 't' we are about to process 
            # is the START of a new episode? 
            # No, usually done[t] means the step 't' RESULTED in termination.
            # CleanRL convention: `done` passed in is the done flag of the PREVIOUS step.
            
            # Mask state using the done flag from the specific timestep
            mask = (1.0 - done[t]).view(1, -1, 1)
            h = h * mask
            c = c * mask
            
            # Forward one step
            input_t = hidden[t].unsqueeze(0) # (1, batch, input_size)
            _, (h, c) = self.lstm(input_t, (h, c))
            outputs.append(h.view(batch_size, -1)) # squeeze (1, batch, hidden) -> (batch, hidden)
            
        new_hidden = torch.stack(outputs, dim=0) # (seq_len, batch, hidden)
        new_hidden = new_hidden.flatten(0, 1)    # (Total_Steps, Hidden)
        
        return new_hidden, (h, c)

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, next_lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), next_lstm_state


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Validation
    if args.num_envs < args.num_minibatches:
        print(f"WARNING: num_envs ({args.num_envs}) < num_minibatches ({args.num_minibatches}). Adjusting minibatches to {args.num_envs}.")
        args.num_minibatches = args.num_envs

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device)
    # store discrete actions as int64
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.int64, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    
    # LSTM State Init
    next_lstm_state = (
        torch.zeros(1, args.num_envs, 128, device=device, dtype=torch.float32),
        torch.zeros(1, args.num_envs, 128, device=device, dtype=torch.float32),
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # clone the initial LSTM state for minibatch slicing later
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            # store action as integer
            actions[step] = action.to(dtype=torch.int64)
            logprobs[step] = logprob

            real_next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            next_obs = torch.as_tensor(real_next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # compute last value and GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the rollout
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization
        envs_per_batch = args.num_envs // args.num_minibatches
        env_inds = np.arange(args.num_envs) 
        flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, envs_per_batch):
                end = start + envs_per_batch
                mb_env_inds = env_inds[start:end]
                
                # mb_inds contains the flattened indices for the selected envs over time
                mb_inds = flat_inds[:, mb_env_inds].flatten()
                
                # Correctly slice the hidden state
                # shape: (1, envs_per_batch, 128)
                h_0 = initial_lstm_state[0][:, mb_env_inds, :].contiguous()
                c_0 = initial_lstm_state[1][:, mb_env_inds, :].contiguous()

                # pass b_dones[mb_inds] which has length seq_len*batch_size; get_states handles reshaping safely
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    (h_0, c_0), 
                    b_dones[mb_inds], 
                    b_actions.long()[mb_inds]
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

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
