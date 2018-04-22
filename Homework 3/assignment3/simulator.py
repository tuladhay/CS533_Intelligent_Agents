
import os
import sys
import numpy as np


class Simulator(object):


    def __init__(self, mdp, starting_pos='RANDOM'):

        self.mdp = mdp
        self.n_actions = mdp.num_actions
        self.row_len = mdp.num_states / 8
        self.timestep = 0
        self.end_idx = self.row_len*2 - 1
        self._isterminal = False

        if starting_pos == 'RANDOM':

            r = int(np.random.choice(range(self.row_len*8)[::4], 1))
            print "Starting in state: {}".format(r)
            self.starting_state = r

        else:
            self.starting_state = starting_pos

        self.state = self.starting_state
        self.space = self.state / 4 + 1

    def action(self, action, env=None):

        # do a toy park
        if env == "toy":
            self.state += 1
            self._isterminal = True
            return self.mdp.rewards[self.state]

        if not self.is_terminal() or self._isterminal == True:

            # transition vector for that action
            observations = self.mdp.T(self.state, action)
            if action == 0 or action == 2:
                # draw a probability from 0.0 - 1.0,
                #print observations


                p = np.random.choice(self.mdp.num_states, 1, replace=False, p=observations)
                n_state = int(p)

                if action == 0:
                    self.state = n_state
                    self.space = n_state/4+1
                    self.timestep += 1

                if action == 2:
                    #self.state = self.space * 4 + 2
                    self.state = n_state
                    self.timestep += 1
                    self._isterminal = True

            else:
                self._isterminal = True
                return 0

            return self.mdp.rewards[self.state]
        else:
            return 0

    def reset(self):

        self.state = self.starting_state
        self.space = self.state / 4
        self.timestep = 0
        self._isterminal = False

    def check_car(self):

        observations = self.mdp.T(self.state, 2)
        p = np.random.choice(self.mdp.num_states, 1, replace=False, p=observations)

        if float(self.mdp.rewards[int(p)]) <= 0:
            return True
        else:
            return False

    def is_terminal(self):

        if (self.state % 4) == 2:
            print "exacting reward of {} for crashing".format(self.mdp.rewards[self.state])
            self._isterminal = True
            return True
        else:
            return False

    def draw_space(self, reward):

        if not self.is_terminal():
            print         "       B\t     A\n"
            for i in range(self.row_len):

                if self.space == i and i < self.row_len:
                    print "{} X  [ ]\t   [ ]    {}".format(i, (self.row_len-1)-i)

                elif self.space == (i+self.row_len):
                    print "{}    [ ]\tX  [ ]    {}".format(i, (self.row_len-1)-i)

                else:
                    print "{}    [ ]\t   [ ]    {}".format(i, (self.row_len-1)-i)


            print "probability of space being taken = 1/n"
            print "Accumulated reward : {}".format(reward)
        else:
            print "EXITED"
