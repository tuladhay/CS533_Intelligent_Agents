import numpy as np

num_lots = 5  # each aisle

prob_occupied = np.array([0.9, 0.7, 0.5, 0.3, 0.1, ])     # auto assign this based on n
mdp_drive = []

iter = 0
for n in range(0, (8*num_lots - 4), 4):  # Going from this state ***
    row = np.zeros(8 * num_lots)
    if n == 8*num_lots - 4:
        row[0] = prob_occupied[iter]


    row[n + 4] = prob_occupied[iter]
    row[n + 6] = 1 - prob_occupied[iter]
    mdp_drive.append(row)

print()
    # for m in range(8*num_lots): # to this state ***
    #     row[]
