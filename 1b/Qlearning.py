'''
*
*   ===================================================
*       CropDrop Bot (CB) Theme [eYRC 2025-26]
*   ===================================================
*
*  This script is intended to be an Boilerplate for 
*  Task 1B of CropDrop Bot (CB) Theme [eYRC 2025-26].
*
*  Filename:		Qlearning.py
*  Created:		    24/08/2025
*  Last Modified:	24/08/2025
*  Author:		    e-Yantra Team
*  Team ID:		    [ CB_2202 ]
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using ICT (NMEICT)
*
*****************************************************************************************
'''
'''You can Modify the this file,add more functions According to your usage.
   You are not allowed to add any external packges,Beside the included Packages.You can use Built-in Python modules.
'''
import numpy as np
import random
import pickle
import os

class QLearningController:
    def __init__(self, n_states=0, n_actions=0, filename="q_table.pkl"): 
        """
        Initialize the Q-learning controller.

        Parameters:
        - n_states (int): Total number of discrete states the agent can be in.
        - n_actions (int): Total number of discrete actions the agent can take.
        - filename (str): Filename to save/load the Q-table (persistent learning).
        """

        self.n_states = n_states
        self.n_actions = n_actions

        # === Configure your learning rate (alpha) and exploration rate (epsilon) here ===
        self.lr = 0.005        # Learning rate (α) — how fast new info overrides old
        self.gamma = 0.9     # Discount factor (γ) — importance of future rewards

        # Epsilon-greedy parameters (exploration)
        self.epsilon_start = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.epsilon = self.epsilon_start

        self.filename = filename

        # Initialize Q-table with zeros; dimensions: [states x actions]
        self.q_table = np.zeros((n_states, n_actions))

        # Action list: should be populated with your lineFollowers valid actions.The Below is just an Example.
        self.action_list = ["left", "forward","right"]  # Example: 0 = left, 1 = forward, 2 = right

        # Action history for oscillation detection
        self.action_history = []
        self.max_history = 6

        # Mapping of action index to specific commands (e.g., motor speeds)
        self.actions = {}

    def Get_state(self, sensor_data):
        """
        Convert raw sensor data into a discrete state index.
        
        Parameters:
        - sensor_data: Any sensor readings from your environment (e.g., distance sensors).
        
        Returns:
        - state (int): A unique index representing the current state.

        === TODO: Implement your logic to convert sensor_data to a discrete state ===
        """
        # Write Your Logic From here #
        threshold = 0.35  # You may tune this after seeing real data
        binary_data = [
            1 if sensor_data['left_corner'] > threshold else 0,
            1 if sensor_data['left'] > threshold else 0,
            1 if sensor_data['middle'] > threshold else 0,
            1 if sensor_data['right'] > threshold else 0,
            1 if sensor_data['right_corner'] > threshold else 0
        ]  #discretize as 1,0

        if binary_data in ([0,0,1,0,0],[0,1,1,1,0]):
            state = 0  # CENTER
        elif binary_data in ([0,1,1,0,0], [0,1,0,0,0],[1,1,1,1,0]):
            state = 1  # SLIGHT LEFT
        elif binary_data in ([1,1,0,0,0], [1,0,0,0,0],[1,1,1,0,0]):
            state = 2  # STRONG LEFT
        elif binary_data in ([0,0,1,1,0], [0,0,0,1,0],[0,1,1,1,1]):
            state = 3  # SLIGHT RIGHT
        elif binary_data in ([0,0,0,1,1], [0,0,0,0,1],[0,0,1,1,1]):
            state = 4  # STRONG RIGHT
        else:
            state = 5  # LOST

        print(binary_data)
        return state


    def Calculate_reward(self, state, prev_state=None, action=None, prev_action=None, consecutive_center=0):        
        """
        Calculate the reward based on the State.

        Parameters:
        - state: Current State of the Linefollower.

        Returns:
        - reward : The reward for the last action.

        === TODO: Implement your reward function here. 
                  For example, give +1 for going straight, -1 for hitting a wall, etc. ===
        """
        # Write Your Logic From here #
        
        base_rewards = {
            0:  +2.00,   # CENTER
            1:  -0.10,   # SLIGHT_LEFT
            2:  -0.50,   # STRONG_LEFT
            3:  -0.10,   # SLIGHT_RIGHT
            4:  -0.50,   # STRONG_RIGHT
            5:  -2.00    # LOST
        }

        reward = base_rewards.get(state, 0)

        # Update action history
        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)

        oscillation_penalty = 0
        
        # Immediate oscillations
        if prev_action is not None and action is not None:
            if (prev_action == "left" and action == "right") or (prev_action == "right" and action == "left"):
                oscillation_penalty += 0.5

        # old oscillation detection (check last 4-6 actions)
        if len(self.action_history) >= 4:
            switches = 0
            for i in range(len(self.action_history) - 1):
                curr_act = self.action_history[i]
                next_act = self.action_history[i + 1]
                if (curr_act == "left" and next_act == "right") or (curr_act == "right" and next_act == "left"):
                    switches += 1
            
            # Penalize if multiple switches
            if switches >= 2:  
                oscillation_penalty += 0.3 * switches
                
            if len(self.action_history) >= 4:
                recent_actions = self.action_history[-4:]
                left_count = recent_actions.count("left")
                right_count = recent_actions.count("right")

                if left_count >= 2 and right_count >= 2:
                    oscillation_penalty += 0.4

        reward -= oscillation_penalty
                
        #Bonus
        if state == 0 and action == "forward":
            reward += 0.2
            
        if state == 0 and (action == "left" or action == "right"):
            reward -= 0.3  # Discourage unnecessary turns
            

        print(state,reward)
        return reward

    
    def update_q_table(self, state, action, reward, next_state):
        """
        Parameters:
        - state (int): Current state.
        - action (int): Action taken.
        - reward (float): Reward received.
        - next_state (int): State reached after taking the action.

        === TODO: Implement the Q-learning update rule here ===
        """
        # Write Your Logic From here #
        a_idx = self.action_list.index(action)

        # Get current Q value
        old_value = self.q_table[state][a_idx]

        next_max = max(self.q_table[next_state])

        # Q-learning update
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)

        # Write updated value back to table
        self.q_table[state][a_idx] = new_value

    def choose_action(self, state):
        """
        Choose an action to perform based on the current state.

        Uses an epsilon-greedy strategy:
        - With probability epsilon: choose a random action (exploration)
        - Otherwise: choose the best known action (exploitation)

        Parameters:
        - state (int): Current state index.

        Returns:
        - action : The action chosen (from action_list).

        === TODO: Replace with your epsilon-greedy selection logic ===
        """
        # Write Your Logic From here #
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            index = random.randint(0, len(self.action_list) - 1)
        else:
            # Exploit: choose the action with the max Q-value for this state
            index = np.argmax(self.q_table[state])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.action_list[index]

    def reset_action_history(self):
        """Reset the action history (useful at the start of new episodes)"""
        self.action_history = []

    def perform_action(self, action):
        """
        Translate an action into motor commands or robot movement.

        Parameters:
        - action : Action selected by the agent.

        Returns:
        - left_speed : Speed for the left motor.
        - right_speed : Speed for the right motor.

        === TODO: Implement action-to-motor translation logic based on your robot ===
        """
        # Write Your Logic From here #


        if action == "left":
            return -0.0, 0.3      # hard left turn
        elif action == "forward":
            return 0.7, 0.7       # straight
        elif action == "right":
            return 0.3, -0.0      # hard right turn

        else:
            #invalid
            left_motor_speed = 0.0
            right_motor_speed = 0.0

        return left_motor_speed, right_motor_speed

    def save_q_table(self):
        """
        Save the current Q-table and parameters to a file.

        Useful for keeping learned behavior between runs.

        === INSTRUCTIONS: You may Save Additional Thing while saving but do not Remove the the following Parameters ===
        """
        with open(self.filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'n_action': self.n_actions,
                'n_states': self.n_states,
                # Add any additional data you want to save
                'lr': self.lr,
                'gamma': getattr(self, 'gamma', 0.9),
                'action_history': getattr(self, 'action_history', [])
            }, f)

    def load_q_table(self):
        """
        Load the Q-table and parameters from file, if it exists.

        Returns:
        - True if data was loaded successfully, False otherwise.
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data.get('q_table', self.q_table)
            self.epsilon = data.get('epsilon', self.epsilon)    
            self.n_actions = data.get('n_action', self.n_actions)
            self.n_states = data.get('n_states', self.n_states)

            self.lr = data.get('lr', self.lr)
            self.gamma = data.get('gamma', getattr(self, 'gamma', 0.9))
            # Load other data here if needed
            self.action_history = data.get('action_history', [])
            return True
        return False