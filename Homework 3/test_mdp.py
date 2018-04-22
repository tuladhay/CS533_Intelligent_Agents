import sys
import operator
import numpy as np
import matplotlib.pyplot as plt
import itertools, functools
import re
import argparse
""" Grid Layout
    grid[0][0] = num_states
    grid[0][1] = num_actions
"""


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
        train[-1] = [float(a) for a in train[-1].split()]
        print()
    return train, actions


class Parsed(object):
    def __init_(self):
        self.input_file = "MDPtest.txt"
        self.gamma = 1
        self.timesteps = 10
        self.epsilon = None


class MDP(object):

    def __init__(self, args, grid, actions):
        self.args = args
        self.grid = grid
        self.gamma = float(args.gamma)
        self.num_states, self.num_actions = grid[0]
        self.actions = actions
        self.rewards = grid[-1]
        self.Value = [x for x in self.rewards]
        self.print_attrs()
        self.epsilon = 0.0

    def print_attrs(self):
        print "number of states: {}\n".format(self.num_states)
        print "number of possible actions: {}\n".format(self.num_actions)
        print "rewards per state: {}\n".format(self.rewards)

    # Reward of being in a given state, given by value iteration
    def Reward(self, state):
        return self.rewards[state]

    # returns probability of going to state X from state Y
    def T(self, state, action, next_state):
        return self.actions[action][state][next_state]

    """
    Value Iteration algorithm:
    U1(state) = Reward(state)
    Ui+1(state) = Reward(state) = gamma*max(for all next states (T(state, action, next_state)(U(i))))
    computes the utility of each state when considering all next states
    """
    def V(self, state):
        p_actions = []
        max_p, sum_p = 0, 0
        for action in range(self.num_actions):
            sum_p = 0
            p_actions = []
            for next_state in range(self.num_states):
                p_actions.append((self.T(state, action, next_state), action, next_state))
            for p in p_actions:
                sum_p += p[0] * self.Value[p[2]]
            if (sum_p > max_p) or (max_p is 0):
                max_p = sum_p
        return self.gamma*max_p + self.Reward(state)

    """
    Q iterates through the algorithm until the utility update is less than delta
    as the utility of each state is updated, the difference between the old and the
    new utility functions can be taken, this is compared against the delta equation
    """
    def Q(self) :

        # fill in Utility for each state
        max_state = 1
        # for finite horizon, collect intermediate V and Pi
        values, policies = [], []
        for it in range(int(self.args.timesteps)):
            for s in range(it):
                print it
                new_value = [0]*self.num_states
                next_prob = []
                for state in range(self.num_states):
                    state_value = self.V(state)
                    if state_value is not None:
                        max_state = max(max_state, abs(self.Value[state] - state_value))
                        new_value[state] = state_value
                self.Value = new_value
            values.append(self.Value)
            policies.append(self.policy())

        return values, policies     # this is for finite horizon


    """ finds the best policy based on the current utility function
        simply returns the best next state: next state with the highest utility
    """
    def policy(self):
        proto_policy = []
        def argmax(state):
            res = {}
            for action in range(self.num_actions):
                res[action] = 0
                self.p_states = []
                for next_state in range(self.num_states):
                    self.p_states.append((self.T(state, action, next_state), action, next_state))
                for p in self.p_states:
                    res[action] += p[0] * self.Value[p[2]]
            return (max(res.items(), key=operator.itemgetter(1))[0] if res else None)
        for state in range(self.num_states):
            proto_policy.append(argmax(state))
        return proto_policy


if __name__=="__main__":
    args = Parsed() # just for testing
    args.input_file = "MDPtest.txt"
    args.gamma = 1
    args.timesteps = 10
    args.epsilon = 0.0

    grid, actions = load_data(args.input_file)
    # Actions contain all the transitions probabilities related with Action1 and Action2
    # Action[action_number][state][next_state]

    mdp = MDP(args, grid, actions)

    #  If finite horizon
    Utility, Policy = mdp.Q()
    for i in range(10):
        U = ["%.5f" % v for v in Utility[i]]
        P = ["%.5f" % v for v in Policy[i]]
        print "***********************************"
        print "Utility for state {} : {}".format(i, U)
        print "Policy for state {}  : {}\n**************************************".format(i, P)

    print()