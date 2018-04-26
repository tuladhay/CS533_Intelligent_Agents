from classMDP import MDP
from utilities import load_args, load_data
from build_parking_mdp import MDP_Parking, Actions
# Can solve for both finite and infinite horizon cases
# This file is different from the one in HW 2. Basically this has the added feature of infinite horizon

'''
Usage: run from terminal / or from IDE to produce the mdp with options as desired

Args:
    -t timesteps
    -g gamma
    -i input file '.txt'
    -e epsilon
    
    example: python test_mdp.py -i "testMDP.txt" -g 0.99 -e 0.0001 -t 0
    
    Timestep of 0 will result in infinite horizon case


Output:
    prints the value function and policy to screen

'''


if __name__ == "__main__":

    args = load_args()

    grid, actions = load_data(args.input_file)
    # Actions contain all the transitions probabilities related with Action1 and Action2
    # Action[action_number][state][next_state]

    mdp = MDP(args, grid, actions)

    #  If infinite horizon
    if args.timesteps is 0:
        Value, Policy = mdp.value_iteration()

        print("Gamma = " + str(args.gamma) + "\nepsilon = " + str(args.epsilon))
        V = ["%.4f" % v for v in Value]
        P = ["%.4f" % v for v in Policy]
        print "**************************************"
        print("Value : " + str(V))
        print("Policy : " + str(P))

    else:
        # finite horizon case
        Value, Policy = mdp.value_iteration()
        for i in range(10):
            V = ["%.4f" % v for v in Value[i]]
            P = ["%.4f" % v for v in Policy[i]]

            print "Value for state {}   : {}".format(i, V)
            print "Policy for state {}  : {}\n".format(i, P)


    # Print policy with action name
    policy_w_name = []
    for p in Policy:
        if p == 0:
            policy_w_name.append("Park")
        elif p == 1:
            policy_w_name.append("Drive")
        else:
            policy_w_name.append("Exit")

    # for i in range(len(policy_w_name)):
    #     print("Value = " + str(V[i]) + " \tPolicy = " + policy_w_name[i])


    mdp = MDP_Parking()
    mdp.build_mdp()
    labels = []

    for x in range(mdp.num_states-1):
        n = mdp.state_id_to_params[x]
        if n[0] == 0:
            aisle = "A"
        if n[0] == 1:
            aisle = "B"

        row = n[1]

        if n[2] == 0:
            O = "Unoccupied"
        if n[2] == 1:
            O = "Occupied  "

        if n[3] == 0:
            park = "NotParked"
        if n[3] == 1:
            park = "Parked  "

        labels.append([aisle, row, O, park])

    print("\n\nPrinting the value function")
    for l in range(len(labels)):
        print(str(labels[l]) + "\t\t" + str(Value[l]))

    print("\n\nPrinting the corresponding actions")
    for l in range(len(labels)):
        print(str(labels[l]) + "\t\t" + str(policy_w_name[l]))
