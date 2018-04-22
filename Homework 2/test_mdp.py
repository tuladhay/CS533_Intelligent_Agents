import numpy as np
import re
import argparse
from classMDP import MDP


def load_data(path):

    with open(path, 'rb') as f:
        train = f.readlines()
        train = [line.strip('\n') for line in train]
        train = [re.sub(r'[^\x00-\x7f]',r'', line) for line in train]
        train[0] = [int(a) for a in train[0].split(' ')]
        num_states, num_actions = train[0]
        lines = num_actions * num_states + num_actions
        grid = []
        for i in range(1, lines+(num_actions-1)):
            if (i-1) % (num_states+1) is not 0:
                grid.append([float(n) for n in train[i].split(' ')[::4]])
                train[i] = [float(n) for n in train[i].split(' ')[::4]]
        actions = []
        for i in range(num_actions):
            actions.append(grid[(i*num_states):((1+i)*num_states)])
        train = np.array(train)

        # To convert the train[-1] containing rewards into a float
        # if this does not work, try replacing split() with split(' ')
        if isinstance(train[-1], str):
            train[-1] = [float(a) for a in train[-1].split()]
        print()
    return train, actions


# For testing purpose
class Parsed(object):
    def __init_(self):
        self.input_file = "MDPtest.txt"
        self.gamma = 1
        self.timesteps = 10
        self.epsilon = None


# For running from terminal
def load_args():

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--timesteps', default=0, help='horizon length, discarded if discount provided', required=False)
    parser.add_argument('-g', '--gamma', default=1, help='discount factor', required=False)
    parser.add_argument('-i', '--input_file', default='MDP1.txt', help='input file with MDP description', required=True)
    parser.add_argument('-e', '--epsilon', default=None, help='epsilon, or early stopping conditions', required=False)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = Parsed() # just for testing
    args.input_file = "MDP2.txt"
    args.gamma = 1
    args.timesteps = 10
    args.epsilon = 0.0

    #args = load_args()

    grid, actions = load_data(args.input_file)
    # Actions contain all the transitions probabilities related with Action1 and Action2
    # Action[action_number][state][next_state]

    mdp = MDP(args, grid, actions)

    #  If finite horizon
    Utility, Policy = mdp.value_iteration()
    for i in range(10):
        U = ["%.4f" % v for v in Utility[i]]
        P = ["%.4f" % v for v in Policy[i]]

        print "Value for state {}   : {}".format(i, U)
        print "Policy for state {}  : {}\n".format(i, P)

    print()