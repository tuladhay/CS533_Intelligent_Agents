import numpy as np
import argparse
from classMDP import MDP



'''
Usage: from terminal to produce the mdp with options as desired

Args:

Output: an MDP with the filename (.txt) provided in the arguments

'''

class MDP_Parking(MDP):
    def __init__(self):
        # Get necessary arguments from terminal. If run from IDE, it will use default values
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
        self.T = []     # transitions
        self.R = []     # rewards

    def build_mdp(self):
        self.R = np.zeros((self.num_states, 1))
        self.R[self.terminal_state] = 1     # terminal state reward
        state_id_counter = 0

        '''Generate rewards for being in a particular state (lotA/B, spot, O, P)
        
        By default:
        drive_reward = -1, discourage going in circles and never parking
        
        if Park,
            if occupied:
                handicap_reward = -100
                crash_reward = -1000
            else
                (reward set based on formula)
        
        terminal state reward = 1
        
        
        
        # The parking lot looks like below:
        
        lot     A   B
        spot    0   0  
                1   1
                2   2
                3   3
                .   .
                .   .
                .   .
                n   n
                
        each of the (lot, spot) in the grid has 4 internal states
        
        '''
        for lot in range(0, 2):  # parking lot column
            for spot in range(0, self.num_rows):  # parking spot in lot A or B
                for occupied in range(0, 2):  # whether spot is occupied
                    for park in range(0, 2):  # whether you are parked there
                        # set up mapping between state ids and interpretable params
                        self.state_id_to_params[state_id_counter] = (lot, spot, occupied, park)
                        self.state_params_to_id[(lot, spot, occupied, park)] = state_id_counter
                        # set up rewards
                        if park == 0:  # not parked
                            self.R[state_id_counter] = self.drive_reward
                        elif park == 1:
                            if occupied == 1:  # parked in spot that is occupied
                                self.R[state_id_counter] = self.crash_reward
                            else:
                                if spot == 0:  # handicapped
                                    self.R[state_id_counter] = self.handicap_reward
                                else:
                                    # Reward based on how far it is from the store
                                    self.R[state_id_counter] = (self.num_rows - spot) * self.parked_reward_factor

                        state_id_counter += 1

        # initialize transitions
        self.T = []
        for a in range(self.num_actions):
            self.T.append(np.zeros((self.num_states, self.num_states)))

        # set up transitions
        for a in range(self.num_actions):
            for lot in range(0, 2):
                for spot in range(0, self.num_rows):
                    for occupied in range(0, 2):
                        for park in range(0, 2):
                            current_state = self.get_state_id(lot, spot, occupied, park)

                            if a == Actions.PARK:
                                if park == 0:
                                    # park the agent
                                    next_state = self.get_state_id(lot, spot, occupied, 1)
                                    self.T[a][current_state, next_state] = 1

                            elif a == Actions.EXIT:
                                if park == 1:
                                    # exit
                                    next_state = self.terminal_state
                                    self.T[a][current_state, next_state] = 1

                            elif a == Actions.DRIVE:
                                if park == 0:
                                    # determine next location of agent
                                    if lot == 0:    # if in aisle A (left side)
                                        if spot == 0:   # if in handicap parking
                                            next_lot = 1    # go to aisle B
                                            next_spot = 0   # first spot
                                        else:               # if not in handicap parking
                                            next_lot = lot  # stay on the same aisle
                                            next_spot = spot - 1    # move upwards
                                    elif lot == 1:
                                        if spot == self.num_rows - 1:   # if at end of aisle B
                                            next_lot = 0                # go to aisle A
                                            next_spot = spot            # last spot
                                        else:
                                            next_lot = lot              # stay in same lot
                                            next_spot = spot + 1        # move downwards

                                    # next state is either occupied or not
                                    next_state1 = self.get_state_id(next_lot, next_spot, 0, park)
                                    next_state2 = self.get_state_id(next_lot, next_spot, 1, park)

                                    if spot == 0:  # handicap
                                        prob_occupied = 0.01
                                    else:
                                        # probability of spot "occupied" based on how far it is from store
                                        prob_occupied = 1. * (self.num_rows - spot) / self.num_rows

                                    # There are only two states an agent can transition when action = "Drive"
                                    # One is the next state with unoccupied parking space
                                    # Another is next state with occupied parking space
                                    # This probability changes depending on how far the parking spot is from the store

                                    # Probability of transitioning into to unoccupied next state
                                    self.T[a][current_state, next_state1] = 1 - prob_occupied

                                    # Probability of transitioning into occupied next state
                                    self.T[a][current_state, next_state2] = prob_occupied


    def load_args(self):
        parser = argparse.ArgumentParser(description='Create MDP for parking problem.')
        parser.add_argument('-n_row', '--num_rows', help="rows [int] in each aisle", type=int, default=10)
        parser.add_argument('-r_h', '--handicap_reward', type=int, default=100,
                            help="reward [int] for parking in handcapped spot")
        parser.add_argument('-r_c', '--crash_reward', type=int, default=1000,
                            help="reward [int] for parking in occupied spot")
        parser.add_argument('-r_d', '--drive_reward', type=int, default=-1, help="reward for driving/not parking")

        args = parser.parse_args()
        return args

    def get_state_id(self, column, row, occupied, parked):
        return self.state_params_to_id[(column, row, occupied, parked)]

    def get_state_params(self, id):
        if id in self.state_id_to_params:
            (column, row, occupied, parked) = self.state_id_to_params[id]
            return (column, row, occupied, parked)
        else:
            return (-1, -1, -1, -1) # this is the terminal state

    def save_mdp(self, filename):

        with open(filename, 'w') as f:
            f.write('{} {}\n\n'.format(self.num_states, self.num_actions))
            for a in range(self.num_actions):
                matrix = '\n'.join('    '.join('{0:0.8f}'.format(float(lot)) for lot in spot) for spot in self.T[a])
                f.write('{}\n\n'.format(matrix))
            f.write('    '.join(['{0:0.8f}'.format(float(a)) for a in self.R]))
            f.write('\n')


class Actions(object):
    PARK = 0
    DRIVE = 1
    EXIT = 2
    strings = ["Park", "Drive", "Exit"]
    print()


if __name__=="__main__":
    mdp = MDP_Parking()
    mdp.build_mdp()
    mdp.save_mdp('my_MDP.txt')
