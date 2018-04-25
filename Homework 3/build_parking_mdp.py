import numpy as np
import argparse


'''
Usage: from terminal to produce the mdp with options as desired

Args:

Output: an MDP with the filename (.txt) provided in the arguments

'''


class Actions(object):
    PARK = 0
    DRIVE = 1
    EXIT = 2
    strings = ["PARK", "DRIVE", "EXIT"]
    print()


class MDP_Parking(object):
    def __init__(self):
        # Following will be initialized by load_args

        args = self.load_args()

        self.num_rows = args.num_rows
        self.handicap_reward = args.handicap_reward
        self.crash_reward = args.crash_reward
        self.drive_reward = args.drive_reward

        self.parked_reward_factor = 10

        self.state_id_to_params = {}
        self.state_params_to_id = {}

        self.num_states = 8*self.num_rows + 1   # Each enumerates state (Location, Occupied, Park)
        self.num_actions = 3
        self.terminal_state = self.num_states - 1
        self.T = []
        self.R = []

    def build_mdp(self):
        # set up rewards and interpretable state mappings
        self.R = np.zeros((self.num_states, 1))
        self.R[self.terminal_state] = 1
        state_id_counter = 0
        for c in range(0, 2):  # parking lot column
            for r in range(0, self.num_rows):  # parking lot row
                for o in range(0, 2):  # whether spot is occupied
                    for p in range(0, 2):  # whether parked
                        # set up mapping between state ids and interpretable params
                        self.state_id_to_params[state_id_counter] = (c, r, o, p)
                        self.state_params_to_id[(c, r, o, p)] = state_id_counter

                        # set up rewards
                        if p == 0:  # not parked
                            self.R[state_id_counter] = self.drive_reward
                        elif p == 1:
                            if o == 1:  # parked in spot that is occupied
                                self.R[state_id_counter] = self.crash_reward
                            else:
                                if r == 0:  # handicapped
                                    self.R[state_id_counter] = self.handicap_reward
                                else:
                                    self.R[state_id_counter] = (self.num_rows - r) * self.parked_reward_factor

                        state_id_counter += 1

        # initialize transitions
        self.T = []
        for a in range(self.num_actions):
            self.T.append(np.zeros((self.num_states, self.num_states)))

        # set up transitions
        for a in range(self.num_actions):
            for c in range(0, 2):
                for r in range(0, self.num_rows):
                    for o in range(0, 2):
                        for p in range(0, 2):
                            current_state = self.get_state_id(c, r, o, p)

                            if a == Actions.PARK:
                                if p == 0:
                                    # park the agent
                                    next_state = self.get_state_id(c, r, o, 1)
                                    self.T[a][current_state, next_state] = 1

                            elif a == Actions.EXIT:
                                if p == 1:
                                    # exit
                                    next_state = self.terminal_state
                                    self.T[a][current_state, next_state] = 1

                            elif a == Actions.DRIVE:
                                if p == 0:
                                    # determine next location of agent
                                    if c == 0:
                                        if r == 0:
                                            next_c = 1
                                            next_r = 0
                                        else:
                                            next_c = c
                                            next_r = r - 1
                                    elif c == 1:
                                        if r == self.num_rows - 1:
                                            next_c = 0
                                            next_r = r
                                        else:
                                            next_c = c
                                            next_r = r + 1

                                    # next state is either occupied or not
                                    next_state1 = self.get_state_id(next_c, next_r, 0, p)
                                    next_state2 = self.get_state_id(next_c, next_r, 1, p)

                                    if r == 0:  # handicap
                                        prob_occupied = 0.001
                                    else:
                                        prob_occupied = 1. * (self.num_rows - r) / self.num_rows

                                    self.T[a][current_state, next_state1] = 1 - prob_occupied
                                    self.T[a][current_state, next_state2] = prob_occupied

    def load_args(self):
        parser = argparse.ArgumentParser(description='Create MDP for parking problem.')
        parser.add_argument('-n_row', '--num_rows', help="rows [int] in each aisle", type=int, default=10)
        parser.add_argument('-r_h', '--handicap_reward', type=int, default=-100,
                            help="reward [int] for parking in handcapped spot")
        parser.add_argument('-r_c', '--crash_reward', type=int, default=-10000,
                            help="reward [int] for parking in occupied spot")
        parser.add_argument('-r_d', '--drive_reward', type=int, default=-1, help="reward for driving/not parking")

        args = parser.parse_args()
        return args

    def get_state_id(self, column, row, occupied, parked):
        """Gets the state id given the interpretable state params.

        Args:
            column: 0 = A, 1 = B
            row: parking row: 0, 1, 2, ..., num_rows-1
            occupied: 1 = occupied by another vehicle, 0 = otherwise
            parked: 1 = parked, 0 = otherwise

        Returns:
            state id
        """
        return self.state_params_to_id[(column, row, occupied, parked)]

    def get_state_params(self, id):
        """Gets interpretable state params given the state id.

        Args:
            id: state id

        Returns:
            tuple (column, row, occupied, parked)
            or (-1, -1, -1, -1) if terminal state
        """
        if id in self.state_id_to_params:
            (column, row, occupied, parked) = self.state_id_to_params[id]
            return (column, row, occupied, parked)
        else:
            return (-1, -1, -1, -1) # terminal state

    def save_to_file(self, filename):
        """Save MDP to file.

        Args:
            filename: path to MDP file
        """
        with open(filename, 'w') as f:
            f.write('{} {}\n\n'.format(self.num_states, self.num_actions))
            for a in range(self.num_actions):
                matrix = '\n'.join('    '.join('{0:0.8f}'.format(float(c)) for c in r) for r in self.T[a])
                f.write('{}\n\n'.format(matrix))
            f.write('    '.join(['{0:0.8f}'.format(float(a)) for a in self.R]))
            f.write('\n')


if __name__=="__main__":
    mdp = MDP_Parking()
    mdp.build_mdp()
    mdp.save_to_file('my_MDP.txt')
