from keras import layers, Model, Input
from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling2D, MaxPooling3D, Conv2D, Conv3D
from keras.optimizers import RMSprop, Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from memory import *
import random
import yaml
from environment import get_main_file

RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m' 

yaml_file = get_main_file()
with open(f'{yaml_file}.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

LOAD_MODELS = data['load_models']
MODEL_DQN_TARGET_NAME = data['model_names']['dqn_target']
MODEL_DQN_EVAL_NAME = data['model_names']['dqn_eval']
MODEL_DQN_PRE_NAME = data['model_names']['dqn_per']
HIGHER_LIMIT = data['higher_limit']
TYPE_MODEL_NAME = data['model_names']['nn_model']


def get_map_risk(map_risk, i, j, k):
        if 0 < i < len(map_risk) and 0 < j < len(map_risk[0]) and 0 < k < len(map_risk[0][0]):
            return map_risk[i][j][k]
        else:
            return -10
        

# -----------------------------------------------------------------------------------------------------------------------------
# Prioritized Experience Replay
class DQNPer:
    def __init__(self, 
                 n_features, n_actions, obstacles, 
                 model_options: list = [], # To optimize the model, like a grid search
                 meaningful_radius: list = [10, 10, 5]
        ):
        # self.env = env
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = 0.9 # learn rate
        self.gamma = 0.999    # discount rate
        self.epsilon = 0.99  # exploration rate
        self.lr = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dqn_learning_rate = 0.001

        if LOAD_MODELS:
            try:
                self.model = load_model(f'{MODEL_DQN_PRE_NAME}.h5')
            except:
                self.model = self._build_model(self.n_actions, 
                                               self.n_features, 
                                               value_to_choose_model_parameters=model_options,
                                               height=(meaningful_radius[0]*2)+1, 
                                               width=(meaningful_radius[1]*2)+1, 
                                               depth=(meaningful_radius[2]*2)+1)
        else:
            self.model = self._build_model(self.n_actions, 
                                           self.n_features, 
                                           value_to_choose_model_parameters=model_options,
                                           height=(meaningful_radius[0]*2)+1, 
                                           width=(meaningful_radius[1]*2)+1, 
                                           depth=(meaningful_radius[2]*2)+1)
        self.memory = Memory(100)  # PER Memory
        self.batch_size = 64


    def possible_next_steps(self, state, future_steps: int = 2):
        matrix_size = (HIGHER_LIMIT['x'], HIGHER_LIMIT['y'], HIGHER_LIMIT['z'])
        initial_position = state.copy()

        directions = [(0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1)]
        directions_steps = [0, 1, 2, 3, 4, 5]

        def calculate_possible_positions(n_steps) -> dict:
            possible_positions = {}

            def calculate_positions_recursive(position, step, path, step_number):
                if step == n_steps:
                    direction = path[0]
                    if step_number not in possible_positions:
                        possible_positions[step_number] = []
                    possible_positions[step_number].append(position)
                    return

                for direction, ds in zip(directions, directions_steps):
                    new_position = tuple(np.add(position, direction))
                    if all(0 <= coord < size for coord, size in zip(new_position, matrix_size)):
                        calculate_positions_recursive(new_position, step + 1, path + [direction], ds)

            calculate_positions_recursive(initial_position, 0, [], -1)

            return possible_positions

        possible_positions = calculate_possible_positions(future_steps)

        return possible_positions


    def _build_model(
            self, 
            n_actions: int, 
            n_features: int,
            # TODO put these values in the yaml, these are the radius values 
            height: int = 21, 
            width: int = 21, 
            depth: int = 11,
            value_to_choose_model_parameters: list = []
        ):
        state_input = Input(shape=(n_features,))
        goal_input = Input(shape=(n_features,))
        map_input = Input(shape=(height, width, depth))
        
        conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(map_input)
        pool1 = layers.MaxPooling2D(pool_size=(1, 1))(conv1)
        flattened_map = layers.Flatten()(pool1)

        # flattened_map = layers.Flatten()(pooled_map)

        concatenated_input = layers.Concatenate()([state_input, goal_input, flattened_map])

        x = layers.Dense(64, activation='relu')(concatenated_input)
        x = layers.Dense(32, activation='relu')(x)
        output_tensor = layers.Dense(n_actions)(x)
        model = Model([state_input, goal_input, map_input], output_tensor)
        model.compile(optimizer=RMSprop(learning_rate=0.1), loss='mse')
        
        return model


    def memorize(self, state, env_map_, goal_input, action, reward, next_state, done, _env_type: str = '', _env_number: int = 0):
        # Calculate TD-Error for Prioritized Experience Replay
        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)

        if _env_type is not '':
            # env_map_ = env_map[np.newaxis, :, :]
            model_name = TYPE_MODEL_NAME + '_' + _env_type + '_' + str(_env_number)
            mem_model = load_model(f'expert_nn_models/{model_name}.h5')

            if self.epsilon > 0.5:
                predict_next_state = np.amax(mem_model.predict([next_state, goal_input], verbose=0)[0])
                predict_current_state = np.amax(mem_model.predict([state, goal_input], verbose=0)[0])
            else:
                predict_next_state = np.amax(self.model.predict([next_state, goal_input, env_map_], verbose=0)[0])
                predict_current_state = np.amax(self.model.predict([state, goal_input, env_map_], verbose=0)[0])

            print('memorize - current state', state, predict_current_state)
            print('memorize - next state', next_state, predict_next_state)
            print('\n')

            td_error = self.alpha * (reward + self.gamma * predict_next_state - predict_current_state)
        else:
            td_error = self.alpha * reward
        
        # Save TD-Error into Memory
        self.memory.add(td_error, (state, env_map_, goal_input, action, reward, next_state, done))


    def choose_action(self, state, goal_input, env_map_):
        random_value = np.random.rand()
        print('EPSILON:', self.epsilon)

        if random_value <= self.epsilon:
            action = random.randrange(self.n_actions)
            print(YELLOW + f'random, action: {action}' + END)
        else:
            state = np.reshape(state, (1, self.n_features))
            act_values = self.model.predict([state, goal_input, env_map_], verbose=0)
            action = np.argmax(act_values[0])  # returns action (Exploitation)

            _next_state = state[0]
            out_limits = False

            if action == 0:
                _next_state[1] += 1
                if _next_state[1] > 41:
                    out_limits = True
            if action == 1:
                _next_state[1] -= 1
                if _next_state[1] < 0:
                    out_limits = True
            if action == 2:
                _next_state[0] += 1
                if _next_state[0] > 41:
                    out_limits = True
            if action == 3:
                _next_state[0] -= 1
                if _next_state[0] < 0:
                    out_limits = True
            if action == 4:
                _next_state[2] += 1
                if _next_state[2] > 5:
                    out_limits = True
            if action == 5:
                _next_state[2] -= 1
                if _next_state[2] < 0:
                    out_limits = True

            if out_limits:
                act_values[0][action] = float('-inf')
                action = np.argmax(act_values[0])

            print(RED + f'network, action: {action}' + END)

        # print('action:', action)
        return action


    def learn(self, _env_type: str = '', _env_number: int = 0):
        print('*****LEARNING*****')
        batch, idxs, is_weight = self.memory.sample(self.batch_size)

        for i in range(self.batch_size):
            state, env_map_, goal_input, action, reward, next_state, done = batch[i]

            if _env_type is not '':
                print(state, next_state, is_weight[i])

                model_name = TYPE_MODEL_NAME + '_' + _env_type + '_' + str(_env_number)
                mem_model = load_model(f'expert_nn_models/{model_name}.h5')
                
                if self.epsilon > 0.5:
                    predict_next_state = np.amax(mem_model.predict([next_state, goal_input], verbose=0)[0])
                else:
                    predict_next_state = np.amax(self.model.predict([next_state, goal_input, env_map_], verbose=0)[0])

                # print('reward', reward, '- action', action, '- weigth', is_weight[i])

                target = self.alpha * (reward + self.gamma * predict_next_state)
                # print('target', target)

                target_f = self.model.predict([state, goal_input, env_map_], verbose=0)

                # print('new target', target_f[0][action] + target)
                # print('\n')
                
                rounded_list = [np.round(np.array(sublist), 2).tolist() for sublist in target_f.tolist()]
                target_f[0][action] += target
            else:
                target_f = np.array([[0,0,0,0,0,0]])
                target_f[0][action] += reward
                
            # Gradient Update. Pay attention at the sample weight as proposed by the PER Paper
            
            rounded_list = [np.round(np.array(sublist), 2).tolist() for sublist in target_f.tolist()]
            print('*****', rounded_list, ' - ', np.argmax(target_f))
            
            self.model.fit([state, goal_input, env_map_], target_f, epochs=25, verbose=0, sample_weight=np.array([is_weight[i]]))
            # self.model.fit([state, goal_input, env_map_], target_f, epochs=50, verbose=0)
        if self.epsilon > self.epsilon_min: # Epsilon Update
            self.epsilon *= self.epsilon_decay
    

    def get_model(self):
        warning_phrase = 'Get model from a loaded one' if LOAD_MODELS else 'The model was created now'
        print(warning_phrase)
        
        return self.model
