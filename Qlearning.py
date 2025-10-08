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
        self.lr = 0  # Learning rate: how much new information overrides old
        self.epsilon = 0  # Exploration rate: chance of choosing a random action

        self.filename = filename

        # Initialize Q-table with zeros; dimensions: [states x actions]
        self.q_table = np.zeros((n_states, n_actions))

        # Action list: should be populated with your lineFollowers valid actions.The Below is just an Example.
        self.action_list = [0, 1, 2]  # Example: 0 = left, 1 = forward, 2 = right

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
        state = None  # Replace this with your own state computation



        return state

    def Calculate_reward(self, state):
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
        reward = 0  # Replace this with your reward logic
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
        # Find index of action in action list
        a_idx = self.action_list.index(action)


        self.q_table[state][a_idx] = 1  # Placeholder: replace with correct update

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
        index = None  # Just a placeholder

        return self.action_list[index]  # Currently returns fixed action (for testing)

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
        left_motor_speed = 1
        right_motor_speed = 1
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
                'n_states': self.n_states
                # Add any additional data you want to save
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
            # Load other data here if needed
            return True
        return False
