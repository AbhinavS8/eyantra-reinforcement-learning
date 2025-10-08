import numpy as np
import random

# --------------------------------------------------------------------------
# ## 1. The GridWorld Environment
# --------------------------------------------------------------------------
class GridWorld:
    """
    A 4x4 grid world environment for a Q-learning agent.
    - S: Start state
    - G: Goal state (reward +10)
    - X: Wall/obstacle (reward -1)
    - O: Open space (reward 0)
    
    Grid Layout:
    [['O', 'O', 'O', 'G'],
     ['O', 'X', 'O', 'O'],
     ['O', 'O', 'O', 'O'],
     ['S', 'O', 'O', 'O']]
    """
    def __init__(self):
        # Rewards: 10 for goal, -1 for wall, 0 otherwise
        self.grid = np.array([
            [0, 0, 0, 10],   # Goal at (0, 3)
            [0, -10, 0, 0],   # Wall at (1, 1)
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.start_state = (3, 0)
        self.state = self.start_state

    def reset(self):
        """Resets the agent to the starting state."""
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        """Checks if a state is a terminal state (goal or wall)."""
        return self.grid[state] == 10 or self.grid[state] == -1

    def get_next_state(self, state, action):
        """Calculates the next state based on the current state and action."""
        row, col = state
        if action == 0:  # Move up
            row = max(0, row - 1)
        elif action == 1:  # Move right
            col = min(3, col + 1)
        elif action == 2:  # Move down
            row = min(3, row + 1)
        elif action == 3:  # Move left
            col = max(0, col - 1)
        return (row, col)

    def step(self, action):
        """
        Takes an action, updates the agent's state, and returns the
        next state, reward, and a 'done' flag.
        """
        next_state = self.get_next_state(self.state, action)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

# --------------------------------------------------------------------------
# ## 2. The Q-Learning Agent
# --------------------------------------------------------------------------
class QLearningAgent:
    """
    An agent that learns to navigate the GridWorld using Q-learning.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        # Q-table: 4x4 grid, 4 possible actions (Up, Right, Down, Left)
        self.q_table = np.zeros((4, 4, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        - Explores by choosing a random action with probability epsilon.
        - Exploits by choosing the best known action otherwise.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """Updates the Q-value for a state-action pair using the Bellman equation."""
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        self.q_table[state][action] = new_q

# --------------------------------------------------------------------------
# ## 3. Training the Agent
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 5000  # Number of training episodes

    print("Training the agent...")
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

    print("Training finished.")
    
# --------------------------------------------------------------------------
# ## 4. Displaying the Learned Policy
# --------------------------------------------------------------------------
#DISPLAY FUNCTION NOT WORKING PROPERLY, Q TABLE WORKS
    print("\nLearned Policy (Arrows indicate the best action from each state):")


# import nump    
    # Map of actions to arrows for visualization
    actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    policy_grid = np.full((4, 4), ' ', dtype=str)

    for r in range(4):
        for c in range(4):
            state = (r, c)
            if env.grid[state] == 10:
                policy_grid[r, c] = 'G'  # Goal
            elif env.grid[state] == -1:
                policy_grid[r, c] = 'X'  # Wall
            else:
                best_action = np.argmax(agent.q_table[state])
                policy_grid[r, c] = actions_map[best_action]
    
    print(agent.q_table)
    print(policy_grid)

# import numpy as np
# import random

# # Define the gridworld environment
# class GridWorld:
#     def __init__(self):
#         self.grid = np.array([
#             [0, 0, 0, 1],  # Goal at (0, 3)
#             [0, -1, 0, 0],  # Wall with reward -1
#             [0, 0, 0, 0],
#             [0, 0, 0, 0]  # Start at (3, 0)
#         ])
#         self.start_state = (3, 0)
#         self.state = self.start_state

#     def reset(self):
#         self.state = self.start_state
#         return self.state

#     def is_terminal(self, state):
#         return self.grid[state] == 1 or self.grid[state] == -1

#     def get_next_state(self, state, action):
#         next_state = list(state)
#         if action == 0:  # Move up
#             next_state[0] = max(0, state[0] - 1)
#         elif action == 1:  # Move right
#             next_state[1] = min(3, state[1] + 1)
#         elif action == 2:  # Move down
#             next_state[0] = min(3, state[0] + 1)
#         elif action == 3:  # Move left
#             next_state[1] = max(0, state[1] - 1)
#         return tuple(next_state)

#     def step(self, action):
#         next_state = self.get_next_state(self.state, action)
#         reward = self.grid[next_state]
#         self.state = next_state
#         done = self.is_terminal(next_state)
#         return next_state, reward, done

# class QLearningAgent:
#     def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
#         self.q_table = np.zeros((4, 4, 4))  # Q-values for each state-action pair
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate

#     def choose_action(self, state):
#         if random.uniform(0, 1) < self.exploration_rate:
#             return random.randint(0, 3)  # Explore
#         else:
#             return np.argmax(self.q_table[state])  # Exploit

#     def update_q_value(self, state, action, reward, next_state):
#         max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
#         current_q = self.q_table[state][action]
#         # Q-learning formula
#         self.q_table[state][action] = current_q + self.learning_rate * (
#             reward + self.discount_factor * max_future_q - current_q
#         )

# env = GridWorld()
# agent = QLearningAgent()

# episodes = 1000  # Number of training episodes

# for episode in range(episodes):
#     state = env.reset()  # Reset the environment at the start of each episode
#     done = False

#     while not done:
#         action = agent.choose_action(state)  # Choose an action
#         next_state, reward, done = env.step(action)  # Take the action and observe next state, reward
#         agent.update_q_value(state, action, reward, next_state)  # Update Q-values
#         state = next_state  # Move to the next state

