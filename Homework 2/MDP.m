classdef MDP < handle
    % This is a class for a Markov Decision Process
    
    properties
        num_states
        num_actions
        actions
        rewards
        gamma = 1
        Value
        time_steps
    end
    
    
    methods       
        function obj = MDP(num_states, num_actions, actions, rewards, tStep, gamma)
            obj.num_states = num_states;
            obj.num_actions = num_actions;
            obj.actions = actions;
            obj.rewards = rewards;
            obj.Value = rewards;
            obj.time_steps = tStep;
            obj.gamma = gamma;
        end
        
        % Reward in the MDP.txt file is a function of (state, action)
        % However, all the actions have the same reward for the
        % corresponding states. Thus to make it easier, I made the reward a
        % function of only states. Might change it later.
        function r = get_reward(obj, state)
            r = obj.rewards(state);
        end
        
        function t = get_tran_prob(obj, state, action, next_state)
            t = obj.actions(state, next_state, action);
        end
        
        function [value, which_action] = get_value(obj, state)
            max_future_reward = 0;
            
            % Loop over all actions
            for a = 1:obj.num_actions
                future_reward = 0;
                for next_state = 1:obj.num_states
                    % from the given state (parameter), compute T(s,a,s')*V
                    future_reward = future_reward + obj.get_tran_prob(state, a, next_state) * obj.Value(next_state);
                end
                
                if (max_future_reward == 0 || future_reward > max_future_reward)
                    max_future_reward = future_reward;
                    which_action = a;    % since reward is in (state, action) pair
                    % However 'which_action' is not being used currently.
                end
            end % for            
            value = obj.gamma*max_future_reward + obj.get_reward(state);
        end % get_value
        
        function finite_horizon_values = fh_value_iteration(obj)
            % This function iteraties for the number of time-steps provided
            finite_horizon_values = zeros(obj.num_states, obj.time_steps);
            for t = 1:obj.time_steps
                new_value = 0;
                
                %for n = 1:t %why this loop from 1:t?
                
                    for state = 1:obj.num_states
                        best_value = obj.get_value(state);
                        finite_horizon_values(state,t) = best_value;
                        % update the obj.Value
                        obj.Value(state) = best_value;
                    end
                    
                %end %why this loop?
                
            end
        end
        
        function finite_horizon_policy = fh_policy(obj)
            finite_horizon_policy = zeros(obj.num_states, obj.time_steps);
            for t = 1:obj.time_steps
                for state = 1:obj.num_states
                    [~, max_action] = obj.get_value(state);
                    finite_horizon_policy(state, t) = max_action;
                end
            end
        end
        
    end % Methods
    
    
end
