"""
Main agent for DQN
"""
import math
import random
import shutil

import gym
import torch
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm

from agents.base import BaseAgent
from graphs.losses.huber_loss import HuberLoss
from graphs.models.dqn import DQN
from utils.env_utils import CartPoleEnv
from utils.misc import print_cuda_statistics
from utils.replay_memory import ReplayMemory, Transition

cudnn.benchmark = True


class DQNAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        # define models (policy and target)
        self.policy_model = DQN(self.config)
        self.target_model = DQN(self.config)
        # define memory
        self.memory = ReplayMemory(self.config)
        # define loss
        self.loss = HuberLoss()
        # define optimizer
        self.optim = torch.optim.RMSprop(self.policy_model.parameters())

        # define environment
        self.env = gym.make('CartPole-v0').unwrapped
        self.cartpole = CartPoleEnv(self.config.screen_width)

        # initialize counter
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU***** ")

        self.policy_model = self.policy_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Initialize Target model with policy model state dict
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='DQN')

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_episode = checkpoint['episode']
            self.current_iteration = checkpoint['iteration']
            self.policy_model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['episode'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'episode': self.current_episode,
            'iteration': self.current_iteration,
            'state_dict': self.policy_model.state_dict(),
            'optimizer': self.optim.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def select_action(self, state):
        """
        The action selection function, it either uses the model to choose an action or samples one uniformly.
        :param state: current state of the model
        :return:
        """
        if self.cuda:
            state = state.cuda()
        sample = random.random()
        eps_threshold = self.config.eps_start + (self.config.eps_start - self.config.eps_end) * math.exp(
            -1. * self.current_iteration / self.config.eps_decay)
        self.current_iteration += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_policy_model(self):
        """
        performs a single step of optimization for the policy model
        :return:
        """
        if self.memory.length() < self.config.batch_size:
            return
        # sample a batch
        transitions = self.memory.sample_batch(self.config.batch_size)

        one_batch = Transition(*zip(*transitions))

        # create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, one_batch.next_state)), device=self.device,dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in one_batch.next_state if s is not None])

        # concatenate all batch elements into one
        state_batch = torch.cat(one_batch.state)
        action_batch = torch.cat(one_batch.action)
        reward_batch = torch.cat(one_batch.reward)

        state_batch = state_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        curr_state_values = self.policy_model(state_batch)
        curr_state_action_values = curr_state_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Get the expected Q values
        expected_state_action_values = (next_state_values * self.config.gamma) + reward_batch
        # compute loss: temporal difference error
        loss = self.loss(curr_state_action_values, expected_state_action_values.unsqueeze(1))

        # optimizer step
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        return loss

    def train(self):
        """
        Training loop based on the number of episodes
        :return:
        """
        for episode in tqdm(range(self.current_episode, self.config.num_episodes)):
            self.current_episode = episode
            # reset environment
            self.env.reset()
            self.train_one_epoch()
            # The target network has its weights kept frozen most of the time
            if self.current_episode % self.config.target_update == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())

        self.env.render()
        self.env.close()

    def train_one_epoch(self):
        """
        One episode of training; it samples an action, observe next screen and optimize the model once
        :return:
        """
        episode_duration = 0
        prev_frame = self.cartpole.get_screen(self.env)
        curr_frame = self.cartpole.get_screen(self.env)
        # get state
        curr_state = curr_frame - prev_frame

        while(1):
            episode_duration += 1
            # select action
            action = self.select_action(curr_state)
            # perform action and get reward
            _, reward, done, _ = self.env.step(action.item())

            if self.cuda:
                reward = torch.Tensor([reward]).to(self.device)
            else:
                reward = torch.Tensor([reward]).to(self.device)

            prev_frame = curr_frame
            curr_frame = self.cartpole.get_screen(self.env)
            # assign next state
            if done:
                next_state = None
            else:
                next_state = curr_frame - prev_frame

            # add this transition into memory
            self.memory.push_transition(curr_state, action, next_state, reward)

            curr_state = next_state

            # Policy model optimization step
            curr_loss = self.optimize_policy_model()
            if curr_loss is not None:
                if self.cuda:
                    curr_loss = curr_loss.cpu()
                self.summary_writer.add_scalar("Temporal_Difference_Loss", curr_loss.detach().numpy(), self.current_iteration)
            # check if done
            if done:
                break

        self.summary_writer.add_scalar("Training_Episode_Duration", episode_duration, self.current_episode)

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
