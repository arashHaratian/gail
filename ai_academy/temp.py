import tensorflow as tf
from draft.discrim_net import Discrim_net
from draft.policy_net import Policy_net
from draft.value_net import Value_net
from draft.training import *
from maze_env import Maze
from environment import get_obstacles
from bi_a_star_2 import main_bi_astar
import itertools
import random

batch = 256
n_space = 3
n_actions = 6
max_len = 100
num_trajs = batch+10


env_dim = tf.constant([40, 40, 6])

n_features = 40 + 1

## ========================== Setting up the env ==========================
obstacles, obstacles_x, obstacles_y, obstacles_z = get_obstacles()
env = Maze(obstacles)


start_axis_min = 2
start_axis_max = 3
all_starts = [[*t] for t in itertools.product(range(start_axis_min, start_axis_max + 1), repeat = n_space)]
state_inputs_train = [start for start in all_starts if start not in obstacles]
state_inputs_train = tf.constant(random.choices(state_inputs_train, k = num_trajs))

end_state = tf.reshape(env.end_node, (1, -1))
goal_inputs_train = tf.repeat(end_state, num_trajs, 0)


## ========================== Creating the networks ==========================
discrim = Discrim_net(n_actions, n_features) 
policy = Policy_net(n_actions, n_features)
value = Value_net(n_actions, n_features)


## ========================== Collecting the learner trajs ==========================
learner_observations, learner_actions, learner_len, learner_rewards =unroll_traj(state_inputs_train, goal_inputs_train,
                                                                                env, policy,
                                                                                batch, num_trajs, max_len)



S_learner = learner_len.sum()
M_learner = learner_len.max()

learner_obs = tf.zeros((S_learner, M_learner, n_space)).numpy() ## zeros since we mask them in RNN
learner_act = tf.zeros((S_learner, 1)).numpy() ## adding 1 to the dim so that it can be concat later
learner_l = tf.zeros((S_learner), dtype=tf.dtypes.int32).numpy()
cnt = 0

## TODO: needs to be check (just copied and added the last dimension, not sure if it works)
for sample in range(num_trajs):
    for seq_length in range(1, learner_len[sample]+1):
        try:
            learner_obs[cnt, :seq_length, :] = learner_observations[sample, :seq_length, :]
            learner_act[cnt, 0] = int(learner_actions[sample][seq_length-2]) 
            learner_l[cnt] = seq_length
            cnt += 1
        except:
            # print("break with index error in Learner Trajectory")
            break


# idx = learner_l != 0
# learner_obs = learner_obs[idx]
# learner_act = learner_act[idx]
# learner_l = learner_l[idx]

if (learner_l == 0).any():
    raise Exception


state_inputs_unrolled = tf.zeros((S_learner, n_space), dtype = tf.dtypes.int32).numpy()
# goal_inputs_unrolled = tf.zeros((S_learner, n_space), dtype = tf.dtypes.int32).numpy()
for idx, seq_length in enumerate(learner_len):
    state_inputs_unrolled[idx*seq_length:(idx+1)*seq_length, :] = state_inputs_train[idx, :]

    ## Since the goals are the same use  the code after for loop
    # goal_inputs_unrolled[idx*seq_length:(idx+1)*seq_length, :] = goal_inputs_train[idx, :]

## If goals are different, coment this line and use the last line in the above for loop
goal_inputs_unrolled = tf.repeat(end_state, S_learner, 0).numpy()


## ========================== Collecting the expert (b-star) trajs ==========================
expert_observations = tf.zeros((num_trajs, max_len, 3)).numpy()
expert_actions = tf.zeros((num_trajs, max_len)).numpy()
expert_len = tf.zeros((num_trajs), dtype=tf.dtypes.int32).numpy()

for idx, (start_point, end_point) in enumerate(zip(state_inputs_train.numpy(), goal_inputs_train.numpy())):
    traj, action, _ = main_bi_astar(use_yaml = True, options=['rain_forest', 17], 
                                    user_start_node = start_point, user_goal_node = end_point)  ## TODO use_yaml should be false
    traj_len = len(traj)
    expert_observations[idx, :traj_len, :] = traj
    expert_actions[idx, :traj_len - 1] = action
    expert_len[idx] = traj_len


S_expert = expert_len.sum()
M_expert = expert_len.max()

expert_obs = tf.zeros((S_expert, M_expert, n_space)).numpy() ## zeros since we mask them in RNN 
expert_act = tf.zeros((S_expert, 1)).numpy() ## adding 1 to the dim so that it can be concat later(in discrim)
expert_l = tf.zeros((S_expert), dtype=tf.dtypes.int32).numpy()
cnt = 0

## TODO: needs to be check (just copied and added the last dimension, not sure if it works)
for sample in range(num_trajs):
    for seq_length in range(1, expert_len[sample]+1):
        try:
            expert_obs[cnt, :seq_length, :] = expert_observations[sample, :seq_length, :]
            expert_act[cnt, 0] = int(expert_actions[sample][seq_length-1])
            expert_l[cnt] = seq_length
            cnt += 1
        except:
            # print("break with index error in expert Trajectory")
            break


## ========================== Train using the collected trajs ==========================
train(policy, value, discrim,
    env,
    learner_obs, learner_act, learner_l,
    expert_obs, expert_act, expert_l,
    state_inputs_unrolled, goal_inputs_unrolled,
    num_discrim_update = 2, num_gen_update = 6, batch = batch)


o, a, l, s, g = sample_batch(batch,learner_obs,learner_act,learner_l,state_inputs_unrolled, goal_inputs_unrolled)