import argparse
import torch


class Parser:
    def __init__(self):
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", device)

        self.parser = argparse.ArgumentParser(description='Navigation_Goal_Deep_Q_Learning-main')
        ''' Model Selector and utilities '''
        self.parser.add_argument("--verbose", default=1, type=int, help="whether to be verbose or not")
        self.parser.add_argument("--TRAIN", default=1, type=int, help="whether to train or test")
        self.parser.add_argument("--EPS", default=1.0, type=float, help="the starting value of epsilon")
        self.parser.add_argument("--EPS_DECAY", default=0.9999, type=float,
                                 help="controls the rate of exponential decay of epsilon, higher means a slower decay")  # was 9999
        self.parser.add_argument("--MIN_EPS", default=0.0001, type=float, help="the minimum value of epsilon")
        self.parser.add_argument("--GAMMA", default=0.99, type=float, help="discount factor value")
        self.parser.add_argument("--BATCH_SIZE", default=512, type=int,
                                 help="number of transitions random sampled from the replay buffer")  # was 256
        self.parser.add_argument("--num_episodes", default=50000, type=int,
                                 help="number of training episodes")  # was 10000
        self.parser.add_argument("--eval_interval", default=50, type=int,
                                 help="every eval_interval epochs the training model is evaluated")
        self.parser.add_argument("--seed", default=42, type=int,
                                 help="environment seed")
        self.parser.add_argument("--device", default=device, type=str,
                                 help="cuda device")
        self.parser.add_argument("--LEARNING_RATE", default=1e-3, type=float,
                                 help="the learning rate of the Adam optimizer, should decrease (1e-5)")
        self.parser.add_argument("--replay_size", default=100000, type=int,
                                 help="size of the replay memory")
        self.parser.add_argument("--update_target_interval", default=200, type=int,
                                 help="every update-target-interval, target is updated")
        self.parser.add_argument("--TEST_EPISODES", default=500, type=int,
                                 help="number of test episodes")

        self.args = self.parser.parse_args()
