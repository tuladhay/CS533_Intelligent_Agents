import sys
import operator
import numpy as np
import matplotlib.pyplot as plt
import itertools, functools
import re
import argparse
from mdp_builder import Builder
from simulator import Simulator
""" Grid Layout
    grid[0][0] = num_states
    grid[0][1] = num_actions
"""
def load_args():

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--timesteps', default=0, help='horizon length, discarded if discount provided', required=False)
    parser.add_argument('-g', '--gamma', default=0.9, help='discount factor', required=False)
    parser.add_argument('-f', '--input_file', default='parking.txt', help='input file with MDP description')
    parser.add_argument('-e', '--epsilon', default=0.01, type=float, help='epsilon, or early stopping conditions', required=False)
    parser.add_argument('-i', '--intermediate', default=False, type=bool,  help='print out intermeiate policies/value functions while it learns', required=False)
    parser.add_argument('-b', '--build', default=False, type=bool, help='use the parking lot planner')
    parser.add_argument('-s', '--spaces', default=8, type=int, help='number of spaces in each row of the parking lot')
    parser.add_argument('-c', '--rcrash', default=-10, type=int, help='penalty on crashing into another car')
    parser.add_argument('-d', '--rdisabled', default=-5, type=int, help='penalty on parking in a handicapped spot')
    parser.add_argument('-r', '--run_trial', default=False, type=bool, help='try policy with a given starting position')
    parser.add_argument('-q', '--q_learner', default=False, type=bool, help='run Q learning algorithm')

    args = parser.parse_args()
    return args


def load_data(path):

    with open(path, 'rb') as f:

        train = f.readlines()
        train = [line.strip('\n') for line in train]
        train = [re.sub(r'[^\x00-\x7f]',r'', line) for line in train]
        train[0] = [int(a) for a in train[0].split(' ')]
        num_states, num_actions = train[0]
        print num_states, num_actions
        lines = num_actions * num_states + num_actions
        grid = []

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
    return train, actions

def build_mdp(args):

    builder = Builder(args.spaces, 0, args.rcrash, args.rdisabled, './careless.txt')
    builder.build() # generates the mdp into "careless.txt"

class MDP(object):

    def __init__(self, args, grid, actions):
        self.args = args
        self.grid = grid
        self.gamma = float(args.gamma)
        # self.epsilon = 1 - self.gamma
        self.epsilon = self.args.epsilon # I did this
        self.num_states, self.num_actions = grid[0]
        self.actions = actions
        self.rewards = grid[-1]
        if type(self.rewards) is str:
            self.rewards = self.rewards.split(' ')
            self.rewards = map(float, self.rewards)
        self.Utility = [x for x in self.rewards]
        self.print_attrs()
        self.timesteps = int(args.timesteps)
        # if (args.epsilon is 0) and (self.gamma > 0):
        #     self.epsilon = ((1*10**-10)*((1-self.gamma)**2))/(2*(self.gamma**2))
        # else: self.epsilon = float(args.epsilon)

    def print_attrs(self):
        print "number of states: {}\n".format(self.num_states)
        print "number of possible actions: {}".format(self.num_actions)
        print "rewards per state: {}".format(self.rewards)

    def Reward(self, state, action=None):
        return self.rewards[state]

    def T(self, state, action, next_state=None):

        if next_state == None:
            print("None state encountered")
            return self.actions[action][state]

        # returns probability of going to state X from state Y """
        return self.actions[action][state][next_state]

    """ Value Iteration algorithm:
            U1(state) = Reward(state)
            Ui+1(state) = Reward(state) = gamma*max(for all next states (T(state, action, next_state)(U(i))))

            computes the utility of each state when considering all next states
    """
    def util(self, state):
        p_actions = []
        max_p, sum_p = 0, 0
        for action in range(self.num_actions):
            sum_p = 0
            p_actions = []
            for next_state in range(self.num_states):
                p_actions.append((self.T(state, action, next_state), action, next_state))
            for p in p_actions:
                sum_p += p[0] * self.Utility[p[2]]
            if (sum_p > max_p) or (max_p is 0):
                max_p = sum_p
        if self.timesteps > 0:
            return max_p + self.Reward(state)
        else:
            return self.gamma*max_p + self.Reward(state)

    """
    Q iterates through the algorithm until the utility update is less than delta
    as the utility of each state is updated, the difference between the old and the
    new utility functions can be taken, this is compared against the delta equation
    """
    def Q(self) :

        max_state = 1
        if self.timesteps == 0:
            print "searching infinite horizon\n"
            print("Epsilon = " + str(self.epsilon))
            print("Gamma = " + str(self.gamma))

            while max_state > self.epsilon:
                max_state = 0
                new_util = [0]*self.num_states
                next_prob = []
                for state in range(self.num_states):
                    # state_util = self.util(state)
                    new_util[state] = self.util(state)
                    # if state_util is not None:
                    #     max_state = max(max_state, abs(self.Utility[state] - state_util))
                    #     new_util[state] = state_util
                    max_state = max(max_state, abs(self.Utility[state] - new_util[state]))

                self.Utility = new_util
            print("Searching done")

        else:
            print "searching on finite horizon"
            # for finite horizon
            utilities, policies = [], []
            for it in range(self.timesteps):
                for s in range(it):
                    new_util = [0]*self.num_states
                    next_prob = []
                    for state in range(self.num_states):
                        state_util = self.util(state)
                        if state_util is not None:
                            max_state = max(max_state, abs(self.Utility[state] - state_util))
                            new_util[state] = state_util
                    self.Utility = new_util
                if self.args.intermediate:
                    print "INTERMEDIATE\n\n"
                    utilities.append(self.Utility)
                    policies.append(self.policy())

            if self.args.intermediate:
                return utilities, policies
            else:
                return self.Utility, self.policy()

        return self.Utility

    """
    finds the best policy based on the current utility function
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
                    res[action] += p[0] * self.Utility[p[2]]
            return (max(res.items(), key=operator.itemgetter(1))[0] if res else None)
        for state in range(self.num_states):
            proto_policy.append(argmax(state))
        return proto_policy


def RL(sim, mdp, explore=0.5, q_iterations=20000):

    reward = 0
    timesteps = 0
    mean_rewards = []
    qtable = np.zeros((mdp.num_states, mdp.num_actions))
    lr = 0.5
    total_reward = 0
    util = mdp.Q()
    policy = mdp.policy()

    for it in range(q_iterations):
        timesteps += 1
        trajectory = []
        rewards = []
        actions = []
        trajectory.append(sim.state)
        rewards.append(mdp.rewards[sim.state])
        sim.reset()

        while not sim._isterminal and timesteps < 500:

            trajectory.append(sim.state)
            rewards.append(int(mdp.rewards[sim.state]))

            if np.random.uniform() < explore:
                a = int(np.random.choice(mdp.num_actions, 1))
            else:
                a = np.argmax(qtable[sim.state])
            a = int(a)
            actions.append(a)
            sim.action(a)
            sim.draw_space(rewards[-1])
            timesteps += 1

        trajectory.append(sim.state)
        rewards.append(int(mdp.rewards[sim.state]))

        # update Q table
        for i in reversed(range(len(actions))):
            state = trajectory[i]
            action = actions[i]
            new_state = trajectory[i+1]
            reward = rewards[i]
            n_actions = qtable.shape[1]
            value = qtable[state, action]
            rollout = np.max([qtable[new_state, x] for x in range(n_actions)])
            new_value = reward + mdp.gamma * rollout
            update = (new_value - value) * lr

            qtable[state, action] = value+update

    #print qtable
    #sys.exit(0)
    total_reward = []
    for jt in range(q_iterations/10):
        #print jt
        reward = 0
        trajectory = []
        actions = []
        timesteps = 0
        sim.reset()
        while not sim._isterminal and timesteps < 500:
            state = sim.state
            reward += sim.action(np.argmax(qtable[state]))
            timesteps += 1

        total_reward.append(reward)
        mean_rewards.append( np.mean(total_reward) )

    print "Mean Reward for Q learner: {}".format(np.mean(mean_rewards))


def simulate_policy(sim, mdp, policy, it, task="random"):

    total_reward = []
    actions = [0, 2]
    if task == "random":
        for i in range(it):
            print i
            reward = 0
            while not sim._isterminal:
                action = actions[np.random.randint(0, 2)]
                print "action: ", action
                print "state", sim.state, "  spot : ", sim.space, "with reward at state of ", sim.mdp.rewards[sim.state]
                reward += sim.action(action)
                print "reward: ", reward
            total_reward.append(reward)
            sim.reset()

    if task == "toy1":
        for _ in range(it):
            reward = 0
            while not sim._isterminal:
                iscar = sim.check_car()
                if iscar:
                    print "acting 0 - drive"
                    reward += sim.action(0)
                else:
                    print mdp.rewards[mdp.num_states/2+9]
                    if (np.random.choice(2, 1, [.4, .6]) == 0) and mdp.rewards[sim.state+1] < mdp.rewards[mdp.num_states/2+9]:
                        print "acting 0 - drive"
                        reward += sim.action(0)
                    else:
                        print "acting 2 - park"
                        reward += sim.action(2, env="toy")
            total_reward.append(reward)
            sim.reset()

    else:
        for _ in range(it):
            reward = 0
            a = []
            while not sim._isterminal:
                print "chooseing action {}".format(policy[sim.space])
                a.append(policy[sim.space])
                reward += sim.action(policy[sim.space])
                sim.draw_space(reward)
            total_reward.append(reward)
            sim = Simulator(mdp)

        print "trejectory: ", a

    print total_reward
    mean_reward = np.mean(total_reward)
    print "mean reward over {} trials: {}".format(it, mean_reward)

    return mean_reward


if __name__ == '__main__':
    args = load_args()
    # if args.build == True:  # True means use the parking lot planner
    #   build_mdp(args)
    #   sys.exit(0)

    args.input_file = "parking.txt"
    args.run_trial = False
    args.epsilon = 0.01

    grid, actions = load_data(args.input_file)
    mdp = MDP(args, grid, actions)
    #sim = Simulator(mdp)

    if int(args.timesteps) > 0: finite = True
    else: finite = False

    # if args.q_learner:
    #     RL(sim, mdp)

    if finite == False:
      Utility = mdp.Q()
      Policy = mdp.policy()
      U = ["%.4f" % v for v in Utility]
      P = ["%.4f" % v for v in Policy]
      print "**************************************\nPolicy: {}\nValue : {}\n**************************************".format(P, U)
      # if args.run_trial:
      #     policy = [int(float(p)) for p in P]
      #
      #     simulate_policy(sim, mdp, policy, 1000, task='toy1' )
    else:
      print "***********************************"
      Utility, Policy = mdp.Q()
      if args.intermediate:
          for i in range(int(args.timesteps)):
              U = ["%.4f" % v for v in Utility[i]]
              print U
          for i in range(int(args.timesteps)):
              P = ["%.4f" % v for v in Policy[i]]
              print P
      else:
          U = ["%.4f" % v for v in Utility]
          P = ["%.4f" % v for v in Policy]
          print "Finite Utility : {}".format(U)
          print "Finite Policy  : {}\n".format(P)


    print()

