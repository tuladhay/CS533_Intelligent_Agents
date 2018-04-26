import numpy as np
import re
import argparse


def load_data(path):

    with open(path, 'rb') as f:
        block = f.readlines()
        block = [line.strip('\n') for line in block]
        block = [re.sub(r'[^\x00-\x7f]',r'', line) for line in block]
        block[0] = [int(a) for a in block[0].split(' ')]
        num_states, num_actions = block[0]

        print("num_states = " + str(num_states) + ", num_actions = " + str(num_actions))
        lines = num_actions * num_states + num_actions
        grid = []

        for i in range(1, lines+(num_actions-1)):
            if ((i-1) % (num_states+1) is not 0) and (len(block)-1 >= i):
                if block[i] == "":
                    pass
                else:
                    grid.append([float(n) for n in block[i].split()])
                    block[i] = [float(n) for n in block[i].split()]
        actions = []
        for i in range(num_actions):
            actions.append(grid[(i*num_states):((1+i)*num_states)])
        block = np.array(block)

        # To convert the block[-1] containing rewards into a float
        # if this does not work, try replacing split() with split(' ')
        if isinstance(block[-1], str):
            block[-1] = [float(a) for a in block[-1].split()]


        # print()
        # Just to make sure everything was in a list of float
        for l in range(len(block)):
            if isinstance(block[l], str):
                block[l] = [float(a) for a in block[l].split()]

    return block, actions


def load_args():

    parser = argparse.ArgumentParser(description='Given an MDP, calculate Value Function and Policy')
    parser.add_argument('-t', '--timesteps', default=0, help='horizon length, default = 0 (infinite horizon)', required=False)
    parser.add_argument('-g', '--gamma', default=0.9, help='discount factor, default = 0.9', required=False)
    parser.add_argument('-i', '--input_file', default='my_MDP.txt', help='input file for MDP, default MDP1.txt', required=False)
    parser.add_argument('-e', '--epsilon', default=0.000001, help='epsilon, default = 0.000001', required=False)
    args = parser.parse_args()
    return args
