import numpy as np
import re
import argparse
from classMDP import MDP

# Can solve for both finite and infinite horizon cases
# This file is different from the one in HW 2. Basically this has the added feature of infinite horizon

def load_data(path):

    with open(path, 'rb') as f:
        train = f.readlines()
        train = [line.strip('\n') for line in train]
        train = [re.sub(r'[^\x00-\x7f]',r'', line) for line in train]
        train[0] = [int(a) for a in train[0].split(' ')]
        num_states, num_actions = train[0]

        print(num_states, num_actions)
        lines = num_actions * num_states + num_actions
        grid = []
        # for i in range(1, lines+(num_actions-1)):
        #     if (i-1) % (num_states+1) is not 0:
        #         grid.append([float(n) for n in train[i].split(' ')[::4]])
        #         train[i] = [float(n) for n in train[i].split(' ')[::4]]
        for i in range(1, lines+(num_actions-1)):
            if ((i-1) % (num_states+1) is not 0) and (len(train)-1 >= i):
                if train[i] == "":
                    pass
                else:
                    grid.append([float(n) for n in train[i].split()])#[::4]])
                    train[i] = [float(n) for n in train[i].split()]#[::4]]
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
        self.input_file = None
        self.gamma = 1
        self.timesteps = 10
        self.epsilon = None


# For running from terminal
def load_args():

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--timesteps', default=0, help='horizon length, discarded if discount provided', required=False)
    parser.add_argument('-g', '--gamma', default=0.90, help='discount factor', required=False)
    parser.add_argument('-i', '--input_file', default='MDP1.txt', help='input file with MDP description', required=True)
    parser.add_argument('-e', '--epsilon', default=0.01, help='epsilon, or early stopping conditions', required=False)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    # Comment this block to run from terminal
    args = Parsed() # just for testing
    args.input_file = "parking.txt"
    args.gamma = 0.90
    args.timesteps = 0
    args.epsilon = 0.01  # for Infinite horizon case. say that max_s |V(s)-V'(s)| = epsilon.
    # *********************************************************************************************

    # Uncomment to run from terminal
    #args = load_args()

    grid, actions = load_data(args.input_file)
    # Actions contain all the transitions probabilities related with Action1 and Action2
    # Action[action_number][state][next_state]

    mdp = MDP(args, grid, actions)

    #  If finite horizon
    if args.timesteps is 0:
        Value, Policy = mdp.value_iteration()
        U = ["%.4f" % v for v in Value]
        P = ["%.4f" % v for v in Policy]
        print "**************************************\nPolicy: {}\nValue : {}\n**************************************".format(
            P, U)

    else:
        Value, Policy = mdp.value_iteration()
        for i in range(10):
            U = ["%.4f" % v for v in Value[i]]
            P = ["%.4f" % v for v in Policy[i]]

            print "Value for state {}   : {}".format(i, U)
            print "Policy for state {}  : {}\n".format(i, P)

        print()
