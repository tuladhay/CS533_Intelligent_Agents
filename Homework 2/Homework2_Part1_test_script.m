% Test for Homework 2 Part 1
% This is a test script to show that Part 1 of the homework is working,
% without parsing the text input. Thus here, the transition probabilities
% are given.
% 
%
% Look at the Homework2 main file


% i,j is the probability of ending up in state j, if you start on state i,
% and take action A1
A1 = [0.2 0.8 0.0;...
      0.0 0.2 0.8;...
      1.0 0.0 0.0];
  
% i,j is the probability of ending up in state j, if you start on state i,
% and take action A2
A2 = [0.9 0.05 0.05; ...
      0.05 0.9 0.05; ...
      0.05 0.05 0.9];

% j, k is the reward you get for taking action k in state j
R = [-1.0 -1.0; ...
     -1.0 -1.0; ...
     0.0 0.0];
% Selecting one reward for both actions since they are the same
Reward = R(:,1);

% Making a 3D array (state, next_state, action(either1or2)
A = A1;
A(:,:,2) = A2;

% Define time steps
tStep = 20;

%MDP (numstates, numactions, actions, rewards)
mdp = MDP(3, 2, A, Reward, tStep);

% Run value iteration
finite_horizon_values = mdp.fh_value_iteration();

% Find the policy
finite_horizon_policy = mdp.fh_policy();