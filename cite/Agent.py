import random
import torch
import numpy as np
from torch.nn import functional as F


class MolDQN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output
    ):
        super(MolDQN, self).__init__()
        self.fc1 = torch.nn.Linear(num_input, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self,
                 num_input,
                 num_hidden,
                 num_output,
                 device,
                 lr,
                 replay_buffer_size
    ):

        self.device = device
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.dqn, self.target_dqn = (
            MolDQN(num_input, num_hidden, num_output).to(self.device),
            MolDQN(num_input, num_hidden, num_output).to(self.device)
        )

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayMemory(replay_buffer_size)
        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(), lr=lr
        )

    def action_step(self, observations, epsilon_threshold):
        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn(observations.to(self.device))
            action = torch.argmax(q_value).cpu().detach().numpy()

        return action

    def train_step(self, batch_size, gamma, polyak):

        experience = self.replay_buffer.sample(batch_size)
        states_ = torch.stack([S for S, *_ in experience]).to(self.device)

        next_states_ = [S for *_, S, _ in experience]
        q, q_target = self.dqn(states_), torch.stack([self.target_dqn(S.to(self.device)).max(dim=0).values.detach() for S in next_states_])

        rewards = torch.stack([R for _, R, *_ in experience]).reshape((1, batch_size)).to(self.device)
        dones = torch.tensor([D for *_, D in experience]).reshape((1, batch_size)).to(self.device)

        q_target = rewards + gamma * (1 - dones) * q_target
        td_target = q - q_target

        loss = torch.where(
            torch.abs(td_target) < 1.0,
            0.5 * td_target * td_target,
            1.0 * (torch.abs(td_target) - 0.5),
        ).mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return loss
