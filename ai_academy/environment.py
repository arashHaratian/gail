import yaml
import matplotlib.pyplot as plt
import os
import numpy as np


def get_main_file():
    return 'initial_config'
    # return 'read_file'


yaml_file = get_main_file()
with open(f'{yaml_file}.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

EXTRACT_LIST_TO_PLOT = data['exctract_list_to_plot']
HIGHER_LIMIT = data['higher_limit']
LOWER_LIMIT = data['lower_limit']
ENV_TYPE = data['env_type']
ENV_NUMBER = data['env_number']
START_NODE = data['start_node']


def get_obstacles(
        exctract_list_to_plot: bool = EXTRACT_LIST_TO_PLOT, 
        defined_yaml: bool = True, # For train
        options: list = [] # For train, the list should have env_type and number
    ):
    if defined_yaml:
        main_file = get_main_file()
    else:
        main_file = 'read_file'
        ENV_TYPE = options[0]
        ENV_NUMBER = options[1]

    if main_file == 'initial_config':
        obstacles = [[2, 2, 2], [3, 3, 2], [4, 4, 2], [5, 5, 2],
                     [2, 2, 3], [3, 3, 3], [4, 4, 3], [5, 5, 3],
                     [2, 1, 2], [3, 1, 2], [4, 1, 2], [5, 1, 2],
                     [2, 1, 3], [3, 1, 3], [4, 1, 3], [5, 1, 3],
                     [1, 4, 2], [1, 5, 2], [1, 4, 3], [1, 5, 3]
                     ]
        # obstacles = [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        
        if exctract_list_to_plot:
            ox = [o[0] for o in obstacles]
            oy = [o[1] for o in obstacles]
            oz = [o[2] for o in obstacles]
    elif main_file == 'read_file':
        file_path = os.path.expanduser(f'../test_environments/{ENV_TYPE}/env_{ENV_NUMBER}.txt')
        data = np.loadtxt(file_path, delimiter=',')

        ox = data[0]
        oy = data[1]
        try:
            oz = data[2]
        except:
            # If the environment just has 2 dimensions
            ox_base = ox.tolist()
            oy_base = oy.tolist()
            oz_base = [START_NODE['z']] * len(ox)

            ox_env, oy_env = [], []
            for _x in range(HIGHER_LIMIT['y']):
                for _y in range(HIGHER_LIMIT['y']):
                    ox_env.append(_x)
                    oy_env.append(_y)

            oz_blocked = np.concatenate(np.array([[i] * len(ox_env) for i in range(HIGHER_LIMIT['z']) if i != START_NODE['z']]), axis=0)
            ox_env *= HIGHER_LIMIT['z'] - 1
            oy_env *= HIGHER_LIMIT['z'] - 1
            
            ox = np.concatenate([ox_base, ox_env], axis=0)
            oy = np.concatenate([oy_base, oy_env], axis=0)
            oz = np.concatenate([oz_base, oz_blocked], axis=0)

        obstacles = []
        for _x, _y, _z in zip(ox, oy, oz):
            obstacles.append([round(_x, 2), round(_y, 2), round(_z, 2)])
            # obstacles.append([round(_x, 2), round(_y, 2)])
    else:
        print('Environment is not defined')

    if exctract_list_to_plot:
        return obstacles, ox, oy, oz
    else:
        return obstacles, [], [], []
    

if __name__ == '__main__':
    _, obstacles_x, obstacles_y, obstacles_z = get_obstacles()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.set_title('Final Trajectory')
    ax.scatter(obstacles_x, obstacles_y, obstacles_z, c='k', marker='s')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(LOWER_LIMIT['x'], HIGHER_LIMIT['x'])
    ax.set_ylim(LOWER_LIMIT['y'], HIGHER_LIMIT['y'])
    ax.set_zlim(LOWER_LIMIT['z'], HIGHER_LIMIT['z'])
    plt.show()
