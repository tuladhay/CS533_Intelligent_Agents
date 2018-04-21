import numpy as np

# This is the MDP class with useful functions:
# Reward(), T(), V() and Q()

class MDP(object):

    def __init__(self, args, grid, actions):
        # grid is only being used for num_states and num_actions and rewards
        # Maybe rename "grid" to something else
        self.args = args
        self.grid = grid
        self.gamma = float(args.gamma)
        self.num_states, self.num_actions = grid[0]
        self.actions = actions
        self.rewards = grid[-1]
        self.Value = [x for x in self.rewards]

    def get_reward(self, state):
        return self.rewards[state]
        # In the simple test example we have three states, and three corresponding rewards
        # This function will just return the reward corresponding to the state
        # The reward as a function of state is static thus just [-1 -1 0] in the test case

    def get_transition_prob(self, action, state, next_state):
        return self.actions[action, state, next_state]

    def V(self, state):
