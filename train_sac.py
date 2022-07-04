import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net_model.sac_net import Action_net
from net_model.sac_net import Critic_net
import traceback


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Pendulum-v1",
                        help="the id of the gym environment") # 此SAC算法针对连续动作空间故需要选择gym上的连续动作环境即box类型对象
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="sac_algorithm",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-after", type=int, default=1000,
                        help="the number of mini-batches")
    parser.add_argument("--update-every", type=int, default=1,
                        help="the number of mini-batches")
    parser.add_argument("--start-epochs", type=int, default=10000,
                        help="the K epochs to update the policy")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="batch size")

    parser.add_argument("--buffer-size", type=int, default=300000,
                        help="buffer size")

    args = parser.parse_args()

    # fmt: on
    return args


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, env, size=100000):
        self.obs_buf = torch.zeros(args.buffer_size, env.observation_space.shape[0]).to(
            device
        )  # concatenate to a tensor num_step*num_env*observation_space
        self.obs2_buf = torch.zeros(
            args.buffer_size, env.observation_space.shape[0]
        ).to(
            device
        )  # concatenate to a tensor num_step*num_env*observation_space
        self.act_buf = torch.zeros(args.buffer_size, env.action_space.shape[0]).to(
            device
        )
        self.rew_buf = torch.zeros(args.buffer_size).to(device)
        self.done_buf = torch.zeros(args.buffer_size).to(device)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=2048):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


if __name__ == "__main__":
    args = parse_args()
    time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{time_now}"
    if args.track:
        import wandb  # visualize tool

        wandb.init(
            project="sac_algorithm",
            entity="long-x",
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(args.gym_id)

    actor = Action_net(env, name="actor").to(device)
    critic1 = Critic_net(env, name="critic1").to(device)
    critic2 = Critic_net(env, name="critic2").to(device)
    target_critic1 = Critic_net(env, name="target_critic1").to(device)
    target_critic2 = Critic_net(env, name="target_critic2").to(device)
    # ensure the paras of Q_target as same as paras of critic
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    # Freeze target networks with respect to optimizers (only update via soft update)
    for p in target_critic1.parameters():
        p.requires_grad = False
    for p in target_critic2.parameters():
        p.requires_grad = False
    # optimizer setting
    optimizer_actor = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer_critic1 = optim.Adam(
        critic1.parameters(), lr=args.learning_rate, eps=1e-5
    )
    optimizer_critic2 = optim.Adam(
        critic2.parameters(), lr=args.learning_rate, eps=1e-5
    )
    # alpha setting
    log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
    log_alpha.requires_grad = True
    log_alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    target_entropy = -np.prod(env.action_space.shape[0])

    # replay buffer
    replay_buffer = ReplayBuffer(env, args.buffer_size)

    # train loop

    try:

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        replay_size = 0
        start_time = time.time()
        obs = torch.Tensor(env.reset(seed=args.seed)).to(device)
        episodic_return = []
        ret = 0

        for update in range(1, args.total_timesteps + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.total_timesteps
                lrnow = frac * args.learning_rate
                optimizer_actor.param_groups[0]["lr"] = lrnow
                optimizer_critic1.param_groups[0]["lr"] = lrnow
                optimizer_critic2.param_groups[0]["lr"] = lrnow

            global_step += 1
            replay_size += 1

            with torch.no_grad():
                if global_step > args.start_epochs:
                    action, _ = actor.get_action(obs)
                    action = action.reshape(-1)

                else:
                    action = torch.from_numpy(env.action_space.sample()).to(device)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if replay_size == args.buffer_size else done
            reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor(
                int(done), dtype=torch.float
            ).to(device)
            # print(next_done)
            replay_buffer.store(obs, action, reward, next_obs, next_done)
            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            ret += reward
            obs = next_obs
            # End of trajectory handling
            if next_done or (replay_size == args.buffer_size):
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                print(global_step)
                print("episodic_return=", ret.item())
                writer.add_scalar("charts/episodic_return", ret, global_step)
                episodic_return.append(ret)
                obs, ret, replay_size = torch.Tensor(env.reset()).to(device), 0, 0

            # Update handling
            if update >= args.update_after and update % args.update_every == 0:
                for j in range(args.update_every):
                    batch = replay_buffer.sample_batch()
                    b_obs = batch["obs"]
                    b_obs2 = batch["obs2"]
                    b_actions = batch["act"]
                    b_rewards = batch["rew"].reshape(-1)
                    b_dones = batch["done"].reshape(-1)
                    with torch.no_grad():
                        # calculate td_target

                        next_actions, next_log_probs = actor.get_action(b_obs2)
                        next_entropy = -next_log_probs.reshape(-1)

                        q1_target_value = target_critic1.get_value(b_obs2, next_actions)
                        q2_target_value = target_critic2.get_value(b_obs2, next_actions)
                        next_value = (
                            torch.min(q1_target_value, q2_target_value)
                            + log_alpha.exp() * next_entropy
                        )
                        td_target = b_rewards + args.gamma * next_value * (1 - b_dones)

                    # Critic loss
                    critic1_loss = torch.mean(
                        F.mse_loss(critic1.get_value(b_obs, b_actions), td_target)
                    )
                    critic2_loss = torch.mean(
                        F.mse_loss(critic2.get_value(b_obs, b_actions), td_target)
                    )
                    # critic1 backward
                    optimizer_critic1.zero_grad()
                    critic1_loss.backward()
                    optimizer_critic1.step()
                    # critic2 backward
                    optimizer_critic2.zero_grad()
                    critic2_loss.backward()
                    optimizer_critic2.step()

                    # update policy net
                    new_actions, log_probs = actor.get_action(b_obs)
                    entropy = -log_probs.reshape(-1)
                    q1_value = critic1.get_value(b_obs, new_actions)
                    q2_value = critic2.get_value(b_obs, new_actions)
                    actor_loss = torch.mean(
                        -log_alpha.exp() * entropy - torch.min(q1_value, q2_value)
                    )
                    # actor backward
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()
                    # update alpha
                    alpha_loss = torch.mean(
                        (entropy - target_entropy).detach() * log_alpha.exp()
                    )
                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    with torch.no_grad():  # soft update target net
                        for param_target, param in zip(
                            target_critic1.parameters(), critic1.parameters()
                        ):
                            param_target.data.copy_(
                                param_target.data * (1.0 - args.tau)
                                + param.data * args.tau
                            )
                        for param_target, param in zip(
                            target_critic2.parameters(), critic2.parameters()
                        ):
                            param_target.data.copy_(
                                param_target.data * (1.0 - args.tau)
                                + param.data * args.tau
                            )

                    writer.add_scalar(
                        "losses/critic1_loss", critic1_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/critic2_loss", critic2_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )

                    # print(
                    #     "SPS:", int(global_step / (time.time() - start_time))
                    # )  # SPS = Samples per second
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

        env.close()
        writer.close()

    except Exception as e:
        print(
            "****************************************************************************************************************************************************"
        )
        print("error in Training")
        print(
            "****************************************************************************************************************************************************"
        )
        print("Reason:")
        print(
            "****************************************************************************************************************************************************"
        )
        traceback.print_exc()
        print(
            "****************************************************************************************************************************************************"
        )
        # print(transition_dict_debug)

    finally:
        actor.save_model()
        critic1.save_model()
        critic2.save_model()
        env.close()
        writer.close()
        print(
            "****************************************************************************************************************************************************"
        )
        print("over")
        print(
            "****************************************************************************************************************************************************"
        )
