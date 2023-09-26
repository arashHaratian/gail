import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers
from bi_a_star_2 import main_bi_astar
import yaml
from environment import get_main_file

HIDDEN_SIZE_ACTOR = 50
HIDDEN_SIZE_CRITIC = 50
HIDDEN_SIZE_VDB = 50
Z_SIZE_VDB = 4


def build_actor(
            n_actions: int, 
            n_features: int,
            height: int = 21, 
            width: int = 21, 
            depth: int = 11
        ):
    state_input = Input(shape=(n_features,))
    goal_input = Input(shape=(n_features,))
    map_input = Input(shape=(height, width, depth))
    
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(map_input)
    pool1 = layers.MaxPooling2D(pool_size=(1, 1))(conv1)
    flattened_map = layers.Flatten()(pool1)

    concatenated_input = layers.Concatenate()([state_input, goal_input])

    x = layers.Dense(128, activation='relu')(concatenated_input)
    x = Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(n_actions)(x)
    model = Model([state_input, goal_input], output_tensor)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    # model.compile(optimizer=RMSprop(learning_rate=0.1), loss='mse')
    
    return model


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


num_inputs = 3  
num_actions = 6  # Actions: front, back, right, legt, up, down

yaml_file = get_main_file()
with open(f'{yaml_file}.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

LOAD_MODELS = data['load_models']
SAVE_MODELS = data['save_models']
TYPE_MODEL_NAME = data['model_names']['nn_model']
HIGHER_LIMIT = data['higher_limit']
LOWER_LIMIT = data['lower_limit']
GOAL_NODE = data['goal_node']
START_NODE = data['start_node']

env_types = ['rain_forest', 'disaster_3d', 'mine_1']
env_numbers = range(0, 10)
for _env_type in env_types:
    for _env_number in env_numbers:
        model_name = TYPE_MODEL_NAME + '_' + _env_type + '_' + str(_env_number)
        if _env_type not in ['3D_tubes', 'disaster_3d']:
            _d3 = False
            GOAL_NODE['z'] = START_NODE['z']
        else:
           _d3 = True

        for _i in range(1000):
            try:
                if _i == 0:
                    _start_node = [START_NODE['x'], START_NODE['y'], START_NODE['z']]
                    _goal_node = [GOAL_NODE['x'], GOAL_NODE['y'], GOAL_NODE['z']]
                else:
                    _start_node = [
                        np.random.randint(LOWER_LIMIT['x'] + 1, HIGHER_LIMIT['x'] / 2), 
                        np.random.randint(LOWER_LIMIT['y'] + 1, HIGHER_LIMIT['y'] / 2), 
                        np.random.randint(LOWER_LIMIT['z'] + 1, HIGHER_LIMIT['z'] / 2)
                    ]
                    _goal_node = [
                        np.random.randint(LOWER_LIMIT['x'] + (HIGHER_LIMIT['x'] / 2) + 1, HIGHER_LIMIT['x']), 
                        np.random.randint(LOWER_LIMIT['y'] + (HIGHER_LIMIT['y'] / 2) + 1, HIGHER_LIMIT['y']), 
                        np.random.randint(LOWER_LIMIT['z'] + (HIGHER_LIMIT['z'] / 2) + 1, HIGHER_LIMIT['z'])
                    ]

                meaningful_radius = np.asarray([5, 5, 2])
                if LOAD_MODELS:
                    try:
                        print('LOAD MODEL')
                        actor_model = load_model(f'expert_nn_models/{model_name}.h5')
                    except:
                        actor_model = build_actor(num_actions, 
                                                num_inputs, 
                                                height=(meaningful_radius[0]*2)+1, 
                                                width=(meaningful_radius[1]*2)+1, 
                                                depth=(meaningful_radius[2]*2)+1
                                                )

                path_one_step, steps, obstacles = main_bi_astar(
                    use_yaml=False, 
                    defined_yaml=False, 
                    options=[_env_type, _env_number], 
                    show_animation=False, 
                    dimension_3= _d3,
                    user_start_node = _start_node,
                    user_goal_node = _goal_node
                )

                goal_input = [_goal_node[0], _goal_node[1], _goal_node[2]]
                env_map = np.zeros((HIGHER_LIMIT['x'], HIGHER_LIMIT['y'], HIGHER_LIMIT['z']))

                for coord in obstacles:
                    x, y, z = coord
                    env_map[int(x), int(y), int(z)] = 1

                state_inputs_train = []
                goal_inputs_train = []
                map_inputs_train = []
                y_train_action = []
                for _cur_state, _step in zip(path_one_step, steps):
                    around_env = get_meaningful_around_env(env_map, _cur_state, meaningful_radius)[np.newaxis, :, :]
                    state_inputs_train.append(_cur_state)
                    goal_inputs_train.append(goal_input)
                    map_inputs_train.append(around_env[0])

                    y_result = [0,0,0,0,0,0]
                    y_result[_step] = 1
                    y_train_action.append(y_result)

                print(y_train_action)
                state_inputs_train = np.array(state_inputs_train, dtype=np.float32)
                goal_inputs_train = np.array(goal_inputs_train, dtype=np.float32)
                map_inputs_train = np.array(map_inputs_train, dtype=np.float32)
                y_train_action = np.array(y_train_action, dtype=np.float32)

                y_train_validity = np.random.randint(2, size=len(y_train_action))
                y_train_validity = np.array([1] * len(y_train_action)) 
                y_train_critic = np.zeros_like(y_train_action)

                actor_model.fit([state_inputs_train, goal_inputs_train], y_train_action, epochs=100, batch_size=64, verbose=0)
        
                print(_start_node)
                print(_goal_node)

                if SAVE_MODELS:
                    actor_model.save(f'expert_nn_models/{model_name}.h5')
            except:
                print('***** FAIL *****')
