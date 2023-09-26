from maze_env import Maze
import RL_brain 
import matplotlib.pyplot as plt
import numpy as np
from bi_a_star_2 import main_bi_astar
import yaml
from environment import get_main_file, get_obstacles
import time

yaml_file = get_main_file()
with open(f'{yaml_file}.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

SHOW_TRAINING_PHASE = data['show_training_phase']
SAVE_MODELS = data['save_models']
MODEL_DQN_TARGET_NAME = data['model_names']['dqn_target']
MODEL_DQN_EVAL_NAME = data['model_names']['dqn_eval']
MODEL_DQN_PRE_NAME = data['model_names']['dqn_per']
INITIAL_REWARD = data['rewards']['initial']
NETWORK = data['network']
START_NODE = data['start_node']
GOAL_NODE = data['goal_node']
HIGHER_LIMIT = data['higher_limit']
LOWER_LIMIT = data['lower_limit']
EPOCHS = data['epochs']
EPISODES = data['episodes']
PLOT_SPEED = data['plot_speed']


def get_meaningful_around_env(env_map: np.array, state: np.array, radius: np.array):
  rx, ry, rz = radius
  meaningful_area = np.ones([(rx*2)+1, (ry*2)+1, (rz*2)+1])
  max_x, max_y, max_z = env_map.shape
  cur_x, cur_y, cur_z = state
  start_x_range, end_x_range = np.clip(cur_x - rx, a_min=0, a_max=max_x), np.clip(cur_x + rx + 1, a_min=0, a_max=max_x)
  start_y_range, end_y_range = np.clip(cur_y - ry, a_min=0, a_max=max_y), np.clip(cur_y + ry + 1, a_min=0, a_max=max_y)
  start_z_range, end_z_range = np.clip(cur_z - rz, a_min=0, a_max=max_z), np.clip(cur_z + rz + 1, a_min=0, a_max=max_z)

  for x_state in range(start_x_range, end_x_range):
    for y_state in range(start_y_range, end_y_range):
      for z_state in range(start_z_range, end_z_range):
        meaningful_area[x_state-start_x_range][y_state-start_y_range][z_state-start_z_range] = env_map[x_state][y_state][z_state]
  
  return meaningful_area


def save_models(dqn):
    if SAVE_MODELS:
        if NETWORK == 'dqn':
            model_target, model_eval = dqn.get_models()
            model_target.save(f'{MODEL_DQN_TARGET_NAME}.h5')
            model_eval.save(f'{MODEL_DQN_EVAL_NAME}.h5')
        elif NETWORK == 'dqn_per':
            model = dqn.get_model()
            model.save(f'{MODEL_DQN_PRE_NAME}.h5')
        else:
            print('Can not save the model')
            print('The network was not configured')


def run(
        dqn, 
        env,
        goal_input: list, 
        env_map: list, 
        obstacles_x: list = [], 
        obstacles_y: list = [], 
        obstacles_z: list = [],
        locate: str = '',
        n_features = [], 
        n_actions = [], 
        obstacles = [],
        _env_type: str = 'rain_forest', 
        _env_number: int = 0
):
    step, step_per = 0, 0
    _complete = 0

    cnt_steps = []
    state = env.reset()

    # positions, a_star_path = bi_a_star.main()
    path_one_step, steps, obstacles = main_bi_astar(
                    use_yaml=False, 
                    defined_yaml=False, 
                    options=[_env_type, _env_number], 
                    show_animation=False, 
                    dimension_3= True,
                    user_start_node = [START_NODE['x'], START_NODE['y'], START_NODE['z']],
                    user_goal_node = [GOAL_NODE['x'], GOAL_NODE['y'], GOAL_NODE['z']]
                )

    # for action, next_state in zip(steps, path_one_step[1:]):
    #     radius = np.asarray([5, 5, 2])
    #     dqn = RL_brain.DQNPer(n_features, n_actions, obstacles, meaningful_radius=radius) 

    #     next_state = np.array(next_state)
    #     env_around = get_meaningful_around_env(env_map, state, radius)[np.newaxis, :, :]
    #     # done if is the last position
    #     _done = True if (next_state == path_one_step[-1]).all() else False
    #     # _done = True if action == a_star_path[-1] and (next_state == positions[-1]).all() else False

    #     print('state:', state, 'next_state:', next_state, 'action:', action)
    #     dqn.memorize(state, env_around, goal_input, action, INITIAL_REWARD, next_state, _done)
    #     state = next_state
    #     dqn.learn()

    #     save_models(dqn)

    if SHOW_TRAINING_PHASE:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for episode in range(EPISODES):
        meaningful_radius = [5, 5, 2]
        dqn = RL_brain.DQNPer(n_features, n_actions, obstacles, meaningful_radius=meaningful_radius) 

        step_per = 0
        print('**************************************************')
        print(f'Episode {episode} - {locate}')
        print('**************************************************')
        state = env.reset()

        # print(env_map)
        # print(env_map.shape)
        # print('**************************************')
        # print(env_map[np.newaxis, :, :])
        # print(env_map[np.newaxis, :, :].shape)

        while True:
            print(f'Episode {episode}')
            radius = np.asarray([5, 5, 2])
            env_around = get_meaningful_around_env(env_map, state, radius)[np.newaxis, :, :]
            action = dqn.choose_action(state.copy(), goal_input, env_around)
            next_state, reward, done = env.step(action)

            dqn.memorize(state, env_around, goal_input, action, reward, next_state, done, _env_type, _env_number)

            if NETWORK == 'dqn':
                if (step > 200) and (step % 5 == 0):
                    dqn.learn()
            if NETWORK == 'dqn_per':
                if (step_per > 100):
                    dqn.learn(_env_type, _env_number)
                    step_per = 0

            state = next_state
            if done: 
                _complete += 1
                break
                
            step += 1
            step_per += 1
            if SHOW_TRAINING_PHASE: plot_training(
                    ax, 
                    state, 
                    _complete, 
                    reward, 
                    obstacles_x, obstacles_y, obstacles_z
            )

        if NETWORK == 'dqn_per':
            # if dqn.memory.tree.n_entries > 1000:
            dqn.learn(_env_type, _env_number)

        save_models(dqn)

        cnt_steps.append(env.get_cnt_steps())

    return cnt_steps


def plot_training(ax, state, complete, reward, obstacles_x, obstacles_y, obstacles_z, dimensions=3):
    if dimensions == 3:
        ax.clear()
        ax.set_title(f'{NETWORK} - Complete episodes: {complete}, last_reward: {round(reward, 2)}')
        ax.scatter(state[0], state[1], state[2], c='r', marker='*')
        ax.scatter(obstacles_x, obstacles_y, obstacles_z, c='k', marker='s')
        ax.scatter([GOAL_NODE['x']], [GOAL_NODE['y']], [GOAL_NODE['z']], c='b', marker='*', label='Goal')
        ax.scatter([START_NODE['x']], [START_NODE['y']], [START_NODE['z']], c='b', marker='^', label='Origin')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(LOWER_LIMIT['x'], HIGHER_LIMIT['x'])
        ax.set_ylim(LOWER_LIMIT['y'], HIGHER_LIMIT['y'])
        ax.set_zlim(LOWER_LIMIT['z'], HIGHER_LIMIT['z'])
        ax.legend()
        plt.pause(PLOT_SPEED)
    else:
        plt.clf()
        plt.plot(state[0], state[1], 'om') 
        plt.plot(obstacles_x, obstacles_y, 'ok')
        plt.plot([GOAL_NODE['x']], [GOAL_NODE['y']], '.r')
        plt.plot([START_NODE['x']], [START_NODE['y']], '.b')
        plt.title(f'{NETWORK} - Complete episodes: {complete}, last_reward: {round(reward, 2)}')
        plt.xlim(LOWER_LIMIT['x'], HIGHER_LIMIT['x'])
        plt.ylim(LOWER_LIMIT['y'], HIGHER_LIMIT['y'])
        plt.pause(PLOT_SPEED)


if __name__ == '__main__':
    # '3D_tubes', 'disaster_3d', 'city_street_periphery', 'city_street_urban', 
    # 'disaster_2d_1', 'disaster_2d_2', 'eucalypte_forest'
    env_types = ['rain_forest', 'disaster_3d', 'mine_1']
    env_numbers = range(0, 10)
    for _env_type in env_types:
        for _env_number in env_numbers:
            if _env_type not in ['3D_tubes', 'disaster_3d']:
                GOAL_NODE['z'] = START_NODE['z']

            locate = f'{_env_type} - {_env_number}'

            obstacles, obstacles_x, obstacles_y, obstacles_z = get_obstacles(
                defined_yaml=False, options=[_env_type, _env_number]
            )

            env = Maze(obstacles)
            n_features, n_actions = env.n_features, env.n_actions

            goal_node = np.array([GOAL_NODE['x'], GOAL_NODE['y'], GOAL_NODE['z']])[np.newaxis]

            # TODO put the np.newaxis here
            env_map = np.zeros((HIGHER_LIMIT['x'], HIGHER_LIMIT['y'], HIGHER_LIMIT['z']))

            for coord in obstacles:
                x, y, z = coord
                env_map[int(x), int(y), int(z)] = 1

            epochs = EPOCHS
            for epoch in range(epochs):
                print(f'epoch {epoch}/{epochs}')
                cnt_steps = run(
                    [], env, goal_node, env_map, 
                    obstacles_x, obstacles_y, obstacles_z, 
                    locate, n_features, n_actions, obstacles,
                    _env_type, _env_number
                )
                print('---------------------------------------')

            plt.show()
