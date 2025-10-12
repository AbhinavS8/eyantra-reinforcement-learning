import numpy as np
import pickle

# Configuration must match Qlearning.py expectations
N_STATES = 6   # states 0..10
N_ACTIONS = 5   # as in action_list from attachment

# Create an initial Q-table (zeros) and reasonable defaults
q_table = np.zeros((N_STATES, N_ACTIONS))

data = {
    'q_table': q_table,
    'epsilon': 0.5,
    'n_action': N_ACTIONS,
    'n_states': N_STATES,
    'lr': 0.1,
    'gamma': 0.99
}

with open('q_table.pkl', 'wb') as f:
    pickle.dump(data, f)

print('Created q_table.pkl with shape', q_table.shape)
