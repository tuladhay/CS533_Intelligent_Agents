% Homework 2
% CS 533
% Intelligent Agents and Decision Making
% Yathartha Tuladhar
% April 16th, 2018

clear; close all;

% Parse the text file:
fileID = fopen('MDP1.txt','r');

StatesActions = textscan(fileID,'%s',2,'Delimiter',' ');
StatesActions = str2double((StatesActions{1}));
num_states = StatesActions(1);
num_actions = StatesActions(2);

% Loop over all actions and create 3D array for actions
% For example, A(:,:,1) will be the dimension for the first action
% Note that A is a transition probability for the states, depending on the
% action (which is the 3rd dimension)
for a=1:num_actions
    T = textscan(fileID,'%f',num_states*num_states,'Delimiter','\t');
    T = cell2mat(T);
    T = reshape(T,num_states,num_states)';
    A(:,:,a)=T;
end

R = textscan(fileID,'%f',num_states*num_actions,'Delimiter','\t');
R = cell2mat(R);
R = reshape(R,num_actions,num_states)';


% Selecting one reward since they are the same
% This makes reward a function of only states R(S)
% TODO: Fix this in the MDP class so that it can access R(S,A)
Reward = R(:,1);

% Define time steps
tStep = 20;

%MDP (numstates, numactions, actions, rewards)
mdp = MDP(num_states, num_actions, A, Reward, tStep);

% Run value iteration
finite_horizon_values = mdp.fh_value_iteration();

% Find the policy
finite_horizon_policy = mdp.fh_policy();