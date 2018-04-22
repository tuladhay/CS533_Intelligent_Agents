import csv
import numpy as np


class Builder(object):

    def __init__(self, n, spots_taken, crash, handicapped, path):

        self.n = n #spots in each row
        self.filled = spots_taken #percentage of spots taken
        self.actions = 3
        self.path = path
        self.crash = crash
        self.handicapped = handicapped

    """
     O = T/F     ; whether the spot is taken or not
     P = T/F     ; whether your car is parking there
     L = 1 .. 2n ; Location index
    """

    def build(self):

        # state order:
        # (L, O, P) ->
        # (i, T, F)  (i, F, T)  (i, T, T)  (i, F, F) -> i+1 -> ...

        with open(self.path, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')

            """ Header of n_states and n_actions """
            writer.writerow([self.n*8, self.actions])
            writer.writerow([])

            """
            Here we need blocks of states
            there will be three main blocks representing each action
            Each block will be n*8 x n*8. This is huge
            Because each parking spot number can be in two rows,
            Each can have different values of O and P
            """
            n = self.n
            scale = range(1,n+1)
            sidx = range(self.n*8)[::4]
            """ DRIVE action """
            # given the drive action, we can always move to the next spot in line
            for i in range(n*8):
                row = [0] * (n*8)
                if i in sidx and (i+4 > (n*8-1)) :
                    idx = (i+4) - (n*8)
                    row[idx] = 1
                elif i in sidx:
                    idx = i+4
                    row[idx] = 1
                writer.writerow(row)
            writer.writerow([])

            """ EXIT action """
            for i in range(n*8):
                writer.writerow([0]*self.n*8)
            writer.writerow([])

            """ PARK action """
            # agent can only park it its local group.
            # were only going to consider states 4 at a time, since they constitute a group
            # There is 0% p of parking in 1 and 4. 3 is a wreck, 2 is open
            # 3 is more probable closer, 2 is more probable when farther from store
            p = 100./self.n
            probs = [p * i for i in range(self.n)]

            #for spot, state in enumerate(range(self.n*8)[::4]):
            spot = 0
            sidx = range(self.n*8)[::4]
            print sidx, len(sidx)
            print probs, len(probs)
            count = 0
            for state in range(self.n*8):

                row = [0]*(self.n*8)
                if (spot == 0) or (spot == self.n): # disabled spot
                    row[state+1] = 0.1
                    row[state+2] = 0.9

                elif state in sidx and state < self.n*4-1: # row B
                    row[state+2] = probs[spot]/100.
                    row[state+1] = 1.- (probs[spot]/100.)

                elif state in sidx and state <= self.n*8-3: # row A
                    print state

                    row[state+2] = probs[spot-self.n-1]/100.
                    row[state+1] = 1.- (probs[spot-self.n-1]/100.)

                if state in sidx:
                    spot += 1
                count += 1
                writer.writerow(row)
                print count

            writer.writerow([])

            """ Reward """
            # everything gets a default of -1 for driving
            reward = [-1]*(self.n*8)
            # every third state tuple is ( _ T T ) -> crash -10
            reward = [self.crash if i in range(2, self.n*8)[::4] else s for i, s in enumerate(reward)]
            # scale positive reward by distance (x, F, T)
            for sx, rx in enumerate(range(1, n*8)[::4]):
                if (sx == 0) or (sx == self.n):
                    reward[rx] = self.handicapped

                # FOR ULTRAMARATHONER
                elif (sx == self.n-1) or (sx == (self.n*2)-1):
                    reward[rx] = 500
                elif rx < 40: # still on row B - states 0 - 4n
                    reward[rx] = (self.n - sx) * 5
                else: # on row A - states 4n - 8n
                    reward[rx] = (self.n - (sx-self.n)) * 5
            # out of the four varients of a state : we want -1, +, -10, -1 reward
            writer.writerow(reward)


