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

    def get_value(self, state):
        p_actions = []  # just a container
        max_p, sum_p = 0, 0
        # need to rename there to more intuitive names
        # is sum_p all the future rewards from that state?
        which_action = None

        for action in range(self.num_actions):
            # For all the actions, we want to find the possible future rewards
            # If an agent takes an action[a], and given it's current "state", calculate the probability to transitioning
            # to all "next_states"
            sum_p = 0
            p_actions = []

            for next_state in range(self.num_states):
                p_actions.append((self.get_transition_prob(action,state,next_state), action, next_state))
                # need double brackets since append only takes one argument in this case

            for p in p_actions:
                sum_p += p[0]*self.Value[p[2]]
                # transition_probability * current_value of that state
                # Now, we have sum of all the future rewards from that state and action, T1V1 + T2V2 + ... TnVn

            if (max_p == 0) or (sum_p > max_p):
                max_p = sum_p
                which_action = action

        # Now, return the value of that state: V(s) = max[ R(s,a) + sum(Tn*Vn) ]
        return (self.gamma*max_p + self.get_reward(state)), which_action


    def value_iteration(self):
        # For finite horizon, iterate till the given timestep, and record the state values for each time step

        values_tstep = [] # stores each iteration values
        policies = []

        for t in range(int(self.args.timesteps)):
            new_value = [0]*self.num_states
            policy_t = [0]*self.num_states

            for s in range(self.num_states):
                new_value[s], policy_t[s] = self.get_value(s)

            self.Value = new_value

            values_tstep.append(new_value)
            policies.append(policy_t)