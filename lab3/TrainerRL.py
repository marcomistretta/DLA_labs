import math
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from DQLN import DQLN
from gym_navigation.memory.replay_memory import ReplayMemory, Transition
from lab3.PPO import PPO


def normalize(state: np.ndarray) -> np.ndarray:
    MAX_DISTANCE = 15.0  # constant
    MIN_DISTANCE = 0.2  # constant
    normalized_state = np.ndarray(shape=state.shape, dtype=np.float64)
    for i in range(len(state)):
        if i < 17:
            normalized_state[i] = (state[i] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        else:
            normalized_state[i] = state[i] / math.pi
    return normalized_state


class TrainerRL:
    def __init__(self, method, writer_path, args):
        self.max_ep_len = 200
        self.args = args

        # environment attributes
        self.writer = SummaryWriter(writer_path)
        self.n_observations = None
        self.n_actions = None
        self.env = None
        self.setup_environment()

        if method == "dqln":
            self.EPS = args.EPS
            self.optimizer = None
            # setup dqln
            self.q_function = DQLN(self.n_observations, self.n_actions).to(self.args.device)
            self.target_q_function = DQLN(self.n_observations, self.n_actions).to(self.args.device)
            self.target_q_function.load_state_dict(self.q_function.state_dict())
            self.replay_buffer = ReplayMemory(args.replay_size)
        elif method == "ppo":
            # setup ppo
            self.ppo_agent = None
            self.init_ppo()
        else:
            print("method not supported, must be either dqln or ppo")
            raise Exception

        self.method = method
        self.print_args()

    def train(self, save_path):
        self.env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
        print()
        print("START Deep Q-Learning Navigation Goal training...")
        if self.args.verbose:
            print("Training with", self.method)

        if self.method == "dqln":
            # Use Adam to optimize.
            self.optimizer = optim.Adam(self.q_function.parameters(), lr=self.args.LEARNING_RATE, weight_decay=1e-5)
            for i_episode in range(0, self.args.num_episodes, 1):
                self.dqln_train_episode(i_episode)

        elif self.method == "ppo":
            self.ppo_train(save_path)

        self.save(save_path)
        print('Training done!')

    def dqln_train_episode(self, i_episode):
        if self.args.verbose:
            # if i_episode % 100 == 0 or (i_episode % self.args.eval_interval == 0):
            print()
            print("Episode: ", i_episode, "of", self.args.num_episodes, end="   ")
        state_observation, info = self.env.reset()
        state_observation = normalize(state_observation)
        state_observation = torch.tensor(state_observation, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        steps = 0

        # Run one episode.
        while True:
            action, observation, reward, done = self.train_step(state_observation)
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.args.device).unsqueeze(0)

            self.replay_buffer.push(state_observation, action, next_state, reward, done)

            # Move to the next state
            state_observation = next_state
            steps += 1

            if done:
                break

        # Perform one step of the optimization (on the policy network)
        self.optimize_model(i_episode)
        # Epsilon decay
        # Update target network.
        if not i_episode % self.args.update_target_interval:
            policy_net_state_dict = self.q_function.state_dict()
            self.target_q_function.load_state_dict(policy_net_state_dict)

        # Every 50 episodes, validate.
        if not i_episode % self.args.eval_interval:
            self.run_validation(i_episode)

    def train_step(self, state_observation):
        action = self.select_action_epsilon(state_observation)
        observation, reward, terminated, truncated, _ = self.env.step(action.item())

        observation = normalize(observation)  # Normalize in [0,1]

        reward = torch.tensor([reward], device=self.args.device)
        done = terminated or truncated
        return action, observation, reward, done

    def select_action_epsilon(self, state):
        sample = random.random()
        self.EPS = max(self.args.MIN_EPS, self.EPS * self.args.EPS_DECAY)

        if sample > self.EPS:
            with torch.no_grad():
                # return index of action with the best Q value in the current state
                return self.q_function(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.args.device, dtype=torch.long)

    def optimize_model(self, i_episode):
        # Make sure we have enough samples in replay buffer.
        if len(self.replay_buffer) < self.args.BATCH_SIZE:
            # if self.args.verbose:
            # print("Model is experiencing to make sure we have enough samples in replay buffer")
            return

        # Sample uniformly from replay buffer.
        transitions = self.replay_buffer.sample(self.args.BATCH_SIZE)

        # This converts batch-arrays of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Concatenate into tensors for batch update.
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.Tensor(batch.done).to(self.args.device)

        # policy_net computes Q(state, action taken)
        state_action_values = self.q_function(state_batch).gather(1, action_batch)

        # Compute the expected Q values with BELLMAN OPTIMALITY Q VALUE EQUATION:
        # Q(state,action) = reward(state,action) + GAMMA * max(Q(next_state, actions), action)
        expected_state_action_values = reward_batch + self.args.GAMMA * (1 - done_batch) * self.target_q_function(next_state_batch).max(1)[0]

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.writer.add_scalar('loss', loss, i_episode)
        if not i_episode % self.args.eval_interval:
            print('loss:', round(loss.item(), 2), end="   ")
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        self.optimizer.step()

    def setup_environment(self):
        self.env = gym.make('gym_navigation:NavigationGoal-v0', render_mode=None, track_id=1)
        self.env.action_space.seed(self.args.seed)
        state_observation, info = self.env.reset(seed=self.args.seed)
        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n
        # Get the number of state observations
        self.n_observations = len(state_observation)

    @torch.no_grad()
    def test(self, load_path):
        self.env = gym.make('gym_navigation:NavigationGoal-v0', render_mode="human", track_id=1)
        print()
        print("START Deep Q-Learning Navigation Goal testing...")
        self.load(load_path)

        if self.method == "dqln":
            self.dqln_test()
        elif self.method == "ppo":
            self.ppo_test()
        print('Testing done!')

    def init_ppo(self):
        # todo try to optimize hyperparameters
        K_epochs = 80  # update policy for K epochs in one PPO update
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network
        print()
        print("PPO hyperparameters:")
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("Gamma : ", gamma)
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
        print()

        # initialize a PPO agent
        self.ppo_agent = PPO(self.n_observations, self.n_actions, lr_actor, lr_critic, gamma, K_epochs,
                             eps_clip, has_continuous_action_space=False)

    def run_validation(self, i_episode):
        if self.method == "dqln":
            rewards = self.dqln_run_validation(self.q_function)
            self.writer.add_scalar('Reward', np.mean(rewards), i_episode)
            self.writer.add_scalar('Epsilon', self.EPS, i_episode)
            if self.args.verbose:
                print('Reward:', round(np.mean(rewards), 2), end="   ")
                print('Epsilon:', round(self.EPS, 4))
        else:
            raise NotImplementedError

    def dqln_run_validation(self, policy_net, num=10):
        running_rewards = [0.0] * num
        for i in range(num):
            state_observation, info = self.env.reset()
            while True:
                state_observation = normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32,
                                                 device=self.args.device).unsqueeze(0)
                action = policy_net(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, _ = self.env.step(action.item())
                running_rewards[i] += reward

                if terminated or truncated:
                    break
        return running_rewards

    def print_args(self):
        print("Summary writer path:", self.writer.log_dir)
        arg_dict = vars(self.args)
        print("Arguments:")
        for key, value in arg_dict.items():
            print(f"\t{key}: {value}")

    def save(self, save_path):
        if self.args.verbose:
            print("save path: ", save_path)
        if self.method == "dqln":
            torch.save(self.q_function.state_dict(), save_path)
        elif self.method == "ppo":
            self.ppo_agent.save(save_path)

    def load(self, load_path):
        if self.args.verbose:
            print("load path: ", load_path)

        if self.method == "dqln":
            self.q_function.load_state_dict(torch.load(load_path))
        elif self.method == "ppo":
            self.ppo_agent.load(load_path)

    def ppo_train(self, save_path):
        update_timestep = self.max_ep_len * 4  # update policy every n timesteps
        print_freq = self.max_ep_len * 10
        print("max ep len : ", self.max_ep_len)
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO eval frequency : " + str(print_freq) + " timesteps")

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while i_episode <= self.args.num_episodes:

            state, info = self.env.reset()
            state = normalize(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.args.device).unsqueeze(
                0)

            current_ep_reward = 0

            for t in range(1, self.max_ep_len + 1):

                # select action with policy
                action = self.ppo_agent.select_action(state)
                # state, reward, done, _ = self.env.step(action)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # saving reward and is_terminals
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    self.ppo_agent.update()  # todo qua potrei passare il summary writer per aggiungere la loss a tensorboard

                # printing average reward
                if time_step % print_freq == 0:
                    # if i_episode > 0 and (i_episode % self.args.eval_interval == 0):
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                            print_avg_reward))
                    self.writer.add_scalar('Reward', print_avg_reward, i_episode)  # op

                    print_running_reward = 0
                    print_running_episodes = 0

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def ppo_test(self):
        print("--------------------------------------------------------------------------------------------")
        total_test_episodes = self.args.TEST_EPISODES

        test_running_reward = 0

        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            state, info = self.env.reset()
            state = normalize(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.args.device).unsqueeze(
                0)

            for t in range(1, self.max_ep_len + 1):
                action = self.ppo_agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if done:
                    break

            # clear buffer
            self.ppo_agent.buffer.clear()

            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        self.env.close()

        print("============================================================================================")

        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))

        print("============================================================================================")

    def dqln_test(self):
        not_terminated = 0
        success = 0
        state_observation, info = self.env.reset()
        for i_episode in range(self.args.TEST_EPISODES):
            if self.args.verbose:
                print()
                print("Episode: ", i_episode, "of", self.args.TEST_EPISODES, end="   ")
            steps = 0
            while True:
                state_observation = normalize(state_observation)
                state_observation = torch.tensor(state_observation, dtype=torch.float32,
                                                 device=self.args.device).unsqueeze(0)
                action = self.q_function(state_observation).max(1)[1].view(1, 1)
                state_observation, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                steps += 1
                if steps >= self.max_ep_len:
                    not_terminated += 1
                    truncated = True

                if terminated or truncated:
                    if not truncated and reward == 500:
                        success += 1
                    state_observation, info = self.env.reset()
                    break
        self.env.close()
        if self.args.verbose:
            print("Executed " + str(self.args.TEST_EPISODES) + " episodes:\n" + str(success) + " successes\n" + str(
                not_terminated) + " episodes not terminated\n" + str(
                self.args.TEST_EPISODES - (success + not_terminated)) + " failures\n")
