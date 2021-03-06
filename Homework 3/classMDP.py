# This is the MDP class with useful functions:
# Reward(), T(), V() and Q()

class MDP(object):
    # Capable of both finite and infinite horizon
    # for infinite horizon, do not provide timestep or set it to 0

    def __init__(self, args, table, actions):
        # table is only being used for num_states and num_actions and rewards

        self.args = args #input_file, gamma, timesteps, epsilon
        self.table = table
        self.gamma = float(args.gamma)
        self.epsilon = args.epsilon  # make it a required argument, unless there is a way to initialize it
        self.num_states, self.num_actions = table[0]
        self.actions = actions

        self.policy = [0] * self.num_states  # for infinite horizon case
        self.timesteps = int(self.args.timesteps) # 0 for infinite horizon case

        # ************************************************************************************
        # THIS WHOLE HACKISH THING WAS TO FIX THE MDP file format

        # reward_start = (self.num_states*self.num_actions) + self.num_actions + 2
        # select= []
        # for n in range(reward_start, len(table)):
        #     select.append(table[n][1])

        # ************************************************************************************

        self.rewards = table[-1]
        self.Value = [x for x in self.rewards]


    def get_reward(self, state):
        return self.rewards[state]
        # In the simple test example we have three states, and three corresponding rewards
        # This function will just return the reward corresponding to the state
        # The reward as a function of state is static thus just [-1 -1 0] in the test case

    def get_transition_prob(self, action, state, next_state):
        # print(action, state, next_state)
        return self.actions[action][state][next_state]

    def get_value_and_action(self, state):
        # ********************************************************************
        # p_actions = []  # just a container
        # max_p, sum_p = 0, 0
        # # need to rename there to more intuitive names
        # which_action = None
        #
        # for action in range(self.num_actions):
        #     # For all the actions, we want to find the possible future rewards
        #     # If an agent takes an action[a], and given it's current "state", calculate the probability to transitioning
        #     # to all "next_states"
        #     sum_p = 0
        #     p_actions = []
        #
        #     for next_state in range(self.num_states):
        #         p_actions.append((self.get_transition_prob(action,state,next_state), action, next_state))
        #         # need double brackets since append only takes one argument in this case
        #
        #     for p in p_actions:
        #         sum_p += p[0]*self.Value[p[2]]
        #         # transition_probability * current_value of that state
        #         # Now, we have sum of all the future rewards from that state and action, T1V1 + T2V2 + ... TnVn
        #
        #     if (max_p == 0) or (sum_p > max_p):
        #         max_p = sum_p
        #         which_action = action
        #
        # # Now, return the value of that state: V(s) = max[ R(s,a) + sum(Tn*Vn) ]
        # return (self.gamma*max_p + self.get_reward(state)), which_action




        # ************** Other method *************************************************
        which_action = None

        p_actions = []
        max_p, sum_p = 0, 0
        for action in range(self.num_actions):
            sum_p = 0
            p_actions = []
            for next_state in range(self.num_states):
                p_actions.append((self.get_transition_prob(action, state, next_state), action, next_state))
            for p in p_actions:
                sum_p += p[0] * self.Value[p[2]]

            if (sum_p > max_p) or (max_p is 0):
                max_p = sum_p
                which_action = action
        if self.timesteps > 0:
            return max_p + self.get_reward(state), which_action
        else:
            return self.gamma * max_p + self.get_reward(state), which_action
        # *******************************************************************************

    def value_iteration(self):
        # For finite horizon, iterate till the given timestep, and record the state values for each time step

        if self.timesteps == 0:
            # see Sutton and Barto 4.4 Value Iteration Algorithms. delta = |v - V(S)|
            # epsilon is the threshold in this algorithm
            delta = 1
            print("Infinite horizon case since time-step provided was 0")
            while delta > self.epsilon:
                delta = 0
                new_value = [0]*self.num_states
                policy_t = [0]*self.num_states

                for s in range(self.num_states):
                    new_value[s], policy_t[s] = self.get_value_and_action(s)
                    delta = max(delta, abs(self.Value[s] - new_value[s]))

                self.Value = new_value
                self.policy = policy_t

            #print("Infinite horizon done")
            return self.Value, self.policy

        else:

            values_tstep = []   # stores each iteration values
            policies = []

            print("Finite horizon case (since time-step provided.")
            for t in range(int(self.args.timesteps)):
                new_value = [0]*self.num_states
                policy_t = [0]*self.num_states

                for s in range(self.num_states):
                    new_value[s], policy_t[s] = self.get_value_and_action(s)

                self.Value = new_value

                values_tstep.append(new_value)
                policies.append(policy_t)

            return values_tstep, policies
