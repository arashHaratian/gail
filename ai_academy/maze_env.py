import numpy as np
import yaml
from environment import get_main_file, get_obstacles

yaml_file = get_main_file()
with open(f'{yaml_file}.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

HIGHER_LIMIT = data['higher_limit']
LOWER_LIMIT = data['lower_limit']
START_NODE = data['start_node']
GOAL_NODE = data['goal_node']
MAX_REWARD = data['rewards']['max']
UNCERTAINTY_OBSTACLE_REWARD = data['rewards']['uncertainty_obstacle']
OBSTACLE_REWARD = data['rewards']['obstacle']
OUT_REWARD = data['rewards']['out']
STEP_REWARD = data['rewards']['step']

class Maze():
    def __init__(self, obstacles):
        super(Maze, self).__init__()
        self.time = 0
        self.action_space = ['f', 'b', 'l', 'r', 'u', 'd']
        self.n_actions = len(self.action_space)
        self.n_features = 3

        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []

        self.inferior_size_limit = [LOWER_LIMIT['x'], LOWER_LIMIT['y'], LOWER_LIMIT['z']]
        self.superior_size_limit = [HIGHER_LIMIT['x'], HIGHER_LIMIT['y'], HIGHER_LIMIT['z']]
        self.env = np.zeros((self.superior_size_limit[0], self.superior_size_limit[1], self.superior_size_limit[2]))
        self.start_node = [START_NODE['x'], START_NODE['y'], START_NODE['z']]
        self.end_node = [GOAL_NODE['x'], GOAL_NODE['y'], GOAL_NODE['z']]
        self.uncertainty_obstacles = []
        self.cur_pos = self.start_node.copy()
        self.obstacles = obstacles


    def _build_maze(self):
        for coord in self.obstacles:
            self.env[coord[0]][coord[1]][coord[2]] = 1


    def reset(self):
        self.time = 0
        self.cur_pos = self.start_node.copy()

        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []

        return np.array(self.cur_pos)
    

    def plot_heatmap(self, matrix: np.array, title_name: str = ''):
        import matplotlib.pyplot as plt
        # Plot the heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='viridis')

        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set the ticks and tick labels
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
        ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over the data and create the annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w")

        ax.set_xlim(LOWER_LIMIT['x'] - 0.5, HIGHER_LIMIT['x'] - 0.5)
        ax.set_ylim(LOWER_LIMIT['y'] - 0.5, HIGHER_LIMIT['y'] - 0.5)
        ax.set_title(title_name)
        plt.show()


    def step_vectorized(self, state_observations, actions=None):
        # Do the actions
        if actions == None:
          n_actions = self.n_actions
          actions = np.asarray([
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, 0, -1]
             ])
          if type(state_observations) is not np.array:
            new_states = np.array(state_observations)[:, np.newaxis, :].repeat(n_actions, axis=1)
          else:
            new_states = state_observations[:, np.newaxis, :].repeat(n_actions, axis=1)
          
          new_states += actions
        else:
          n_actions = 1
          if type(actions) is not np.array:
            actions = np.asarray(actions)

          if type(state_observations) is not np.array:
            new_states = np.asarray(state_observations)

          new_states += actions

        num_samples = len(state_observations)
        rewards = np.zeros((num_samples, n_actions))
        dones = np.zeros((num_samples, n_actions))

        # Create masks to end, obstacles, and out of bounds
        if n_actions == self.n_actions:
          end_node_match = np.all(new_states == self.end_node, axis=2)
          obstacle_matches = np.all(np.any(new_states[:, :, np.newaxis] == np.array(self.obstacles)[np.newaxis, np.newaxis, :], axis=2), axis=2)
          out_of_bounds = np.any(
              (new_states < self.inferior_size_limit) | (new_states > self.superior_size_limit), axis=2
          )
        else:
          end_node_match = np.all(new_states == self.end_node, axis=1)
          obstacle_matches = np.all(new_states[:, None] == self.obstacles, axis=2).any(axis=1)
          out_of_bounds = np.any(
              (new_states < self.inferior_size_limit) | (new_states > self.superior_size_limit), axis=1
          )

        rewards[end_node_match] = MAX_REWARD
        dones[end_node_match] = 1
        rewards[obstacle_matches] = OBSTACLE_REWARD
        rewards[out_of_bounds] = OUT_REWARD

        # Final mask
        if n_actions == self.n_actions:
          mask = (rewards == 0) & ~end_node_match & ~obstacle_matches
        else:
          mask = (rewards == 0) & ~end_node_match[:, np.newaxis] & ~obstacle_matches[:, np.newaxis]

        rewards[mask] = STEP_REWARD

        # new_states = np.clip(new_states, self.inferior_size_limit, np.array(self.superior_size_limit))

        return new_states, rewards, dones
    

    def step(self, action):
        self.time += 1

        if action == 0:   # front
            self.cur_pos[1] += 1
        elif action == 1:   # back
            self.cur_pos[1] -= 1
        elif action == 2:   # right
            self.cur_pos[0] += 1
        elif action == 3:   # left
            self.cur_pos[0] -= 1
        elif action == 4:   # up
            self.cur_pos[2] += 1
        elif action == 5:   # down
            self.cur_pos[2] -= 1

        # reward function
        if self.cur_pos == self.end_node:
            # reward = max_map_risk * 1.5
            # TODO do this value based in the env size
            # reward = MAX_REWARD / self.time
            reward = MAX_REWARD
            # reward = HIGHER_LIMIT['x'] * HIGHER_LIMIT['y'] * HIGHER_LIMIT['z'] / self.time
            done = True
        elif self.cur_pos in self.obstacles:
            # reward = min_map_risk * 1.3
            reward = OBSTACLE_REWARD
            done = False
        elif self.cur_pos in self.uncertainty_obstacles:
            # reward = min_map_risk * 1.3
            reward = UNCERTAINTY_OBSTACLE_REWARD
            done = False
        elif self.cur_pos[0] < self.inferior_size_limit[0] or self.cur_pos[1] < self.inferior_size_limit[1] or self.cur_pos[0] >= self.superior_size_limit[0] or self.cur_pos[1] >= self.superior_size_limit[1]:
            # reward = min_map_risk * 1.8
            reward = OUT_REWARD
            done = False
        else:
            # map_risk_value = map_risk.T[self.cur_pos[0], self.cur_pos[1]]
            # reward = map_risk_value if map_risk_value < 0 else -1 if map_risk_value == 0 else 0
            reward = STEP_REWARD
            # reward = map_risk[self.cur_pos[0], self.cur_pos[1]]
            done = False

        self.cur_pos[0] = np.clip(self.cur_pos[0], LOWER_LIMIT['x'], HIGHER_LIMIT['x'] - 1)
        self.cur_pos[1] = np.clip(self.cur_pos[1], LOWER_LIMIT['y'], HIGHER_LIMIT['y'] - 1)
        self.cur_pos[2] = np.clip(self.cur_pos[2], LOWER_LIMIT['z'], HIGHER_LIMIT['z'] - 1)

        self.trajectory_x.append(self.cur_pos[0])
        self.trajectory_y.append(self.cur_pos[1])
        self.trajectory_z.append(self.cur_pos[2])

        return np.array(self.cur_pos), reward, done


    def get_final_trajectory(self):
        return self.trajectory_x, self.trajectory_y, self.trajectory_z
    

    def get_cnt_steps(self):
        return self.time
