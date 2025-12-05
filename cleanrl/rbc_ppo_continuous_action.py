# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
    exp_name: str = "rbc_ppo_continuous_action"
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
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
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
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
    rbc_warmup_frac: float = 0.5
    """Fraction of total timesteps to freeze the residual head (Freeze-and-Fine-Tune)"""

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
    def __init__(self, envs, rbc_enabled=True):
        super().__init__()
        self.rbc_enabled = rbc_enabled
        obs_shape = np.array(envs.single_observation_space.shape).prod()

        # Actor Network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        # TD Head (Fast Learner)
        self.critic_td = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Residual Head (Slow Learner)
        if self.rbc_enabled:
            self.critic_res = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
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
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # Get combined value for advantage calculation
        value = self.get_value(x)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, rbc_enabled=args.rbc_enabled).to(device)
    
    # Separate optimizers
    base_params = list(agent.actor_mean.parameters()) + list(agent.actor_logstd) + list(agent.critic_td.parameters())
    optimizer_base = optim.Adam(base_params, lr=args.learning_rate, eps=1e-5)

    if args.rbc_enabled:
        optimizer_res = optim.Adam(
            agent.critic_res.parameters(), 
            lr=args.learning_rate * args.rbc_lr_scale, 
            eps=1e-5
        )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_total = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_td = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_base.param_groups[0]["lr"] = lrnow
            if args.rbc_enabled:
                optimizer_res.param_groups[0]["lr"] = lrnow * args.rbc_lr_scale

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_hat = agent.get_action_and_value(next_obs)
                v_td_comp, _ = agent.get_components(next_obs)
                
                values_total[step] = value_hat.flatten()
                values_td[step] = v_td_comp.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
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
            
            returns = advantages + values_total

        # 2. High-Lambda Return for Residual Head (Low Bias, lambda=0.99)
        if args.rbc_enabled:
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
                    gae_res = delta_res + args.gamma * args.rbc_lambda_res * nextnonterminal * lastgaelam_res
                    lastgaelam_res = gae_res
                    returns_res[t] = gae_res + values_total[t]

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_total.reshape(-1)
        b_values_td = values_td.reshape(-1)
        
        if args.rbc_enabled:
            b_returns_res = returns_res.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        is_residual_active = args.rbc_enabled and (global_step > args.total_timesteps * args.rbc_warmup_frac)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue_hat = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # TD Head Loss
                newvalue_td = new_v_td.view(-1)
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
                loss_base = pg_loss - args.ent_coef * entropy_loss + v_loss_td * args.vf_coef

                optimizer_base.zero_grad()
                loss_base.backward()
                nn.utils.clip_grad_norm_(base_params, args.max_grad_norm)
                optimizer_base.step()

                # Residual Head Loss
                if is_residual_active:
                    newvalue_res = new_v_res.view(-1)
                    target_res = b_returns_res[mb_inds] - new_v_td.detach().view(-1)
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer_base.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_td", v_loss_td.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.rbc_enabled:
            writer.add_scalar("rbc/loss_res", loss_res.item(), global_step)
            writer.add_scalar("rbc/is_active", float(is_residual_active), global_step)
            writer.add_scalar("rbc/residual_magnitude", b_values_td.abs().mean().item(), global_step)
            mean_res = b_values.mean().item() - b_values_td.mean().item()
            mean_td = b_values_td.mean().item()
            writer.add_scalar("rbc/bias_fraction", mean_res / (abs(mean_td) + 1e-6), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=lambda envs: Agent(envs, rbc_enabled=False), # Evaluation doesn't need RBC components
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()