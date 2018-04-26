from classMDP import MDP
from utilities import load_args, load_data

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

        print()
