import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import time
import os


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Action_net(nn.Module):
    def __init__(self, env, resume=False, name=None):
        super(Action_net, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.observation_space.shape[0], 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, env.action_space.shape[0]), std=0.01),
        )
        self.action_bound = env.action_space.high[0]
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_space.shape[0]))
        self.name = name
        if resume:
            self.load_model()

    def save_model(self):
        checkpoint = {
            "net_actor": self.actor_mean.state_dict(),
            # 'optimizer_actor': self.actor_optimizer.state_dict(),
            # 'optimizer_critic': self.critic_optimizer.state_dict(),
        }
        time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        path = os.path.abspath("./checkpoints/sac_" + self.name + time_now + ".pth")

        path_dir = os.path.abspath("./checkpoints")
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        torch.save(checkpoint, path)
        print(
            "****************************************************************************************************************************************************"
        )
        print("save model to {}".format(path))

    def load_model(self):

        path = os.path.abspath("./checkpoints/sac_" + self.name + ".pth")  # 断点路径
        checkpoint = torch.load(path)  # 加载断点
        self.actor_mean.load_state_dict(checkpoint["net_actor"])

        # 加载模型可学习参数
        print(
            "****************************************************************************************************************************************************"
        )
        print("load model from {}".format(path))

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        normal_sample = probs.rsample()

        # print(probs.log_prob(normal_sample))
        log_prob = probs.log_prob(normal_sample).sum(1)
        action = torch.tanh(normal_sample)
        # print(torch.log(1 - torch.tanh(action).pow(2) + 1e-7))

        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7).sum(1)
        action = action * self.action_bound
        return action, log_prob


class Critic_net(nn.Module):
    def __init__(self, env, resume=False, name=None):
        super(Critic_net, self).__init__()

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    env.observation_space.shape[0] + env.action_space.shape[0],
                    256,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.name = name
        if resume:
            self.load_model()

    def save_model(self):
        checkpoint = {
            "net_critic": self.critic.state_dict(),
            # 'optimizer_actor': self.actor_optimizer.state_dict(),
            # 'optimizer_critic': self.critic_optimizer.state_dict(),
        }
        time_now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        path = os.path.abspath("./checkpoints/sac_" + self.name + time_now + ".pth")

        path_dir = os.path.abspath("./checkpoints")
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        torch.save(checkpoint, path)
        print(
            "****************************************************************************************************************************************************"
        )
        print("save model to {}".format(path))

    def load_model(self):

        path = os.path.abspath("./checkpoints/sac_" + self.name + ".pth")  # 断点路径
        checkpoint = torch.load(path)  # 加载断点
        self.critic.load_state_dict(checkpoint["net_critic"])
        # 加载模型可学习参数
        print(
            "****************************************************************************************************************************************************"
        )
        print("load model from {}".format(path))

    def get_value(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.critic(cat)
