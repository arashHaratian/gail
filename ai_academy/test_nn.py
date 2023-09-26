from keras.models import load_model
import numpy as np
from environment import get_obstacles

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

_env_type = 'rain_forest'
_env_number = 0

model_name = 'nn_model' + '_' + _env_type + '_' + str(_env_number)
mem_model = load_model(f'expert_nn_models/{model_name}.h5')

obstacles, _, _, _ = get_obstacles(defined_yaml=False, options=[_env_type, _env_number])

env_map = np.zeros((42, 42, 6))
for coord in obstacles:
    x, y, z = coord
    env_map[int(x), int(y), int(z)] = 1

state = [0, 41, 2]
goal = [41,41,2]
meaningful_radius = [5, 5, 2]
around_env = get_meaningful_around_env(env_map, state, meaningful_radius)

state = np.asarray(state)[np.newaxis]
goal = np.asarray(goal)[np.newaxis]
around_env = np.asarray(around_env)[np.newaxis]

prediction = mem_model.predict([[state], [goal], [around_env]], verbose=0)

print(prediction, np.argmax(prediction))
