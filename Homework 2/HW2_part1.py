import numpy as np
import argparse
import re

def load_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--timesteps', default=10, help='horizon length, discarded if discount provided', required=False)
    parser.add_argument('-g', '--gamma', default=0, help='discount factor', required=False)
    parser.add_argument('-i', '--input_file', default='MDP1.txt', help='input file with MDP description', required=True)
    parser.add_argument('-e', '--epsilon', default=None, help='epsilon, or early stopping conditions', required=False)
    args = parser.parse_args()
    return args

def load_data(path):

    with open(path, 'rb') as f:
        train = f.readlines()
        train = [line.strip('\n') for line in train]
        train = [re.sub(r'[^\x00-\x7f]', r'', line) for line in train] # replace non-ASCII characters with a single space
        train[0] = [int(a) for a in train[0].split(' ')] # turn line 1 into list of [state, actions]
        num_states, num_actions = train[0]
        lines = num_actions * num_states + num_actions
        grid = []
        for i in range(1, lines + (num_actions - 1)):
            if (i - 1) % (num_states + 1) is not 0:
                grid.append([float(n) for n in train[i].split(' ')[::4]])
                train[i] = [float(n) for n in train[i].split(' ')[::4]]
        actions = []
        for i in range(num_actions):
            actions.append(grid[(i * num_states):((1 + i) * num_states)])
        train = np.array(train)
    return train, actions


class MDP(object):
    def __init__(self, args, grid, actions):
        self.args = args
        self.grid = grid
        self.num_states, self.num_actions = grid[0]
        self.actions = actions
        self.rewards = grid[-1]
        self.gamma = 1.0 # MAYBE USE ARGS
        self.Value = [x for x in self.rewards]
        self.print_attrs()
        #if args.epsilon is None:
        #    self.epsilon = ((1 * 10 ** -10) * ((1 - self.gamma) ** 2)) / (2 * (self.gamma ** 2))
        #else:
        #    self.epsilon = float(args.epsilon)
        print()

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

    def V(self, state):
        p_actions = []
        max_p, sum_p = 0,0

        for action in range(self.num_actions):
            sum_p = 0
            p_actions = []

            for next_state in range(self.num_states):
                p_actions.append((self.T(state, action, next_state), action, next_state))

            for p in p_actions:
                sum_p += p[0] * self.Value[p[2]]
            if (sum_p > max_p) or (max_p is 0):
                max_p = sum_p
        return self.gamma * max_p + self.Reward(state)

    def Q(self):

        # fill in Utility for each state
        max_state = 1
        values, policies = [], []
        for it in range(int(self.args.timesteps)):
            for s in range(it):
                print it
                new_value = [0] * self.num_states
                next_prob = []
                for state in range(self.num_states):
                    state_value = self.V(state)
                    if state_value is not None:
                        max_state = max(max_state, abs(self.Value[state] - state_value))
                        new_value[state] = state_value
                self.Value = new_value
            values.append(self.Value)
            policies.append(self.policy())
        #return values, policies
        return self.Value

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
    #args = load_args()
    # grid here is the "return train"
    grid, actions = load_data("MDPtest_old.txt")
    mdp = MDP(1, grid, actions)

    #print(mdp.T(0, 1, 2))
    print()