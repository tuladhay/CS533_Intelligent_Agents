% Homework 2
% CS 533
% Intelligent Agents and Decision Making
% Yathartha Tuladhar
% April 16th, 2018

clear; close all; clc;
%   Which part of the assignment would you like to run?     %
%   Select from 1, 2, 3
%   1: Test MDP
%   2: Own MDP (gridworld)
%   3: Provided MDP

%   I am using a gamma of 0.9 (eventhough this is a finite horizon case
%   This can be changes in the 

% ************************************************* %
PART = 3;
gamma = 0.9;
% ************************************************* %

% Parse the text file:
if PART == 1
    fileID = fopen('MDPtest.txt','r');
elseif PART == 2
    fileID = fopen('Part2_Own_MDP_Gridworld.txt','r');
elseif PART == 3    % CHOOSE WHICH MDP TO RUN
    fileID = fopen('MDP1.txt','r');
    %fileID = fopen('MDP2.txt','r');
else
    disp("Enter Valid Part")
end
        
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
tStep = 10;

%MDP (numstates, numactions, actions, rewards)
mdp = MDP(num_states, num_actions, A, Reward, tStep, gamma);

% Run value iteration
finite_horizon_values = mdp.fh_value_iteration();

% Find the policy
finite_horizon_policy = mdp.fh_policy();

if (PART == 1 || PART == 3)
    disp("finite horizon values: ");
    disp(finite_horizon_values);
    
    disp("finite horizon policy: ");
    disp(finite_horizon_policy);
end

if PART == 2
    policy = finite_horizon_policy(:,tStep);
    policy = reshape(policy, 5, 5);
    
    % res is the resulting policy in plain words
    res(policy==1) = "up";
    res(policy==2) = "down";
    res(policy==3) = "right";
    res(policy==4) = "left";
    res = res';
    res= reshape(res,5,5);
    
    values = finite_horizon_values(:,tStep);
    values = reshape(values, 5, 5);
    
    disp("finite horizon values: ");
    disp(values);
    disp("finite horizon policy: ");
    disp(res)
end %if

