import tensorflow as tf
import numpy as np
from draft.discrim_net import Discrim_net
from draft.policy_net import Policy_net
from draft.value_net import Value_net
from draft.training import *
from maze_env import Maze
from environment import get_obstacles
from bi_a_star_2 import main_bi_astar
import itertools
import random


SEED = 100
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
# tf.config.set_visible_devices([], 'GPU')

batch = 32
n_space = 3
n_actions = 6
max_len = 40
num_trajs = 100

env_option = ['disaster_3d', 17]

env_dim = tf.constant([8, 8, 6])

num_train_iter = 500

n_features = 8

## ========================== Setting up the env ==========================
obstacles, obstacles_x, obstacles_y, obstacles_z = get_obstacles(defined_yaml = False, options = env_option)
env = Maze(obstacles)


start_axis_min = 2
start_axis_max = 5

all_starts = [[*t] for t in itertools.product(range(start_axis_min, start_axis_max + 1), repeat = n_space)]
state_inputs_train = [start for start in all_starts if start not in obstacles]
state_inputs_train = tf.constant(random.choices(state_inputs_train, k = num_trajs))

end_state = tf.reshape(env.end_node, (1, -1))
goal_inputs_train = tf.repeat(end_state, num_trajs, 0)

## ========================== Creating the networks ==========================
initial_weights = tf.keras.initializers.glorot_uniform(seed=SEED)
policy = Policy_net(n_actions, n_features, kernel_initializer=initial_weights, lr = 5e-4)
value = Value_net(n_actions, n_features, kernel_initializer=initial_weights, lr = 5e-4)
discrim = Discrim_net(n_actions, n_features, kernel_initializer=initial_weights, lr = 5e-4)

tf.keras.utils.plot_model(discrim, show_shapes=True,  show_trainable=True)
tf.keras.utils.plot_model(policy, show_shapes=True,  show_trainable=True)
tf.keras.utils.plot_model(value, show_shapes=True,  show_trainable=True)

## TODO: Forcing the weights of embeddings in discrim and value to be the same as policy 


for i in range(num_train_iter):


    all_starts = [[*t] for t in itertools.product(range(start_axis_min, start_axis_max + 1), repeat = n_space)]
    state_inputs_train = [start for start in all_starts if start not in obstacles]
    state_inputs_train = tf.constant(random.choices(state_inputs_train, k = num_trajs))

    end_state = tf.reshape(env.end_node, (1, -1))
    goal_inputs_train = tf.repeat(end_state, num_trajs, 0)


    print(f"Iteration {i}")
    ## ========================== Collecting the learner trajs ==========================
    learner_observations, learner_actions, learner_len, learner_rewards = unroll_traj(state_inputs_train, goal_inputs_train,
                                                                                      env, policy,
                                                                                      batch, num_trajs, max_len)


    ## TODO: 1- with no last state
    learner_len[learner_len == (max_len+1)] -= 1

    ## TODO:2- with no action for the last state
    # learner_len[learner_len != (max_len+1)] += 1

    ## TODO: 3- 3 embeddings for each axis

    print(f"{i} : {tf.reduce_mean(learner_rewards)} ; {(learner_len != (max_len)).mean()}")

    S_learner = learner_len.sum()
    M_learner = learner_len.max()

    learner_obs = tf.zeros((S_learner, M_learner, n_space)).numpy() ## zeros since we mask them in RNN
    learner_act = tf.zeros((S_learner, 1)).numpy() ## adding 1 to the dim so that it can be concat later
    learner_l = tf.zeros((S_learner), dtype=tf.dtypes.int32).numpy()
    cnt = 0
    # old_cnt = 0

    ## TODO: needs to be check (just copied and added the last dimension, not sure if it works)
    for sample in range(num_trajs):
        for seq_length in range(1, learner_len[sample]+1):
            try:
                learner_obs[cnt, :seq_length, :] = learner_observations[sample, :seq_length, :]
                learner_act[cnt, 0] = int(learner_actions[sample][seq_length-1]) 
                learner_l[cnt] = seq_length
                cnt += 1
            except:
                # print(f"break with index error in Learner Trajectory {sample}")
                break
        # learner_act[old_cnt:c~nt-1 , 0] = learner_actions[sample, :seq_length - 1]
        # old_cnt = cnt-1


    idx = learner_l != 0
    learner_obs = learner_obs[idx]
    learner_act = learner_act[idx]
    learner_l = learner_l[idx]

    if (learner_l == 0).any():
        raise Exception


    state_inputs_unrolled = learner_obs[:, 0, :]
    goal_inputs_unrolled = tf.repeat(end_state, S_learner, 0).numpy()

    ## ========================== Collecting the expert (b-star) trajs ==========================
    expert_observations = tf.zeros((num_trajs, max_len + 1, 3)).numpy()
    expert_actions = tf.zeros((num_trajs, max_len)).numpy()
    expert_len = tf.zeros((num_trajs), dtype=tf.dtypes.int32).numpy()

    for idx, (start_point, end_point) in enumerate(zip(state_inputs_train.numpy(), goal_inputs_train.numpy())):
        traj, action, _ = main_bi_astar(use_yaml = False, options=env_option, dimension_3=True, user_start_node = start_point, user_goal_node = end_point)
        
        traj_len = len(traj)
        expert_observations[idx, :traj_len, :] = traj
        expert_actions[idx, :traj_len - 1] = action
        expert_len[idx] = traj_len -1 
        # expert_len[idx] = traj_len


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
                print("break with index error in expert Trajectory")
                break

    if (expert_l == 0).any():
        raise Exception

    ## ========================== Train using the collected trajs ==========================
    policy, value, discrim = train(policy, value, discrim,
                                   env,
                                   learner_obs, learner_act, learner_l,
                                   expert_obs, expert_act, expert_l,
                                   state_inputs_unrolled, goal_inputs_unrolled,
                                   num_discrim_update = 2, num_gen_update = 6, batch = batch*4
    )

    learner_observations, learner_actions, learner_len, learner_rewards = unroll_traj(state_inputs_train, goal_inputs_train,
                                                                                      env, policy,
                                                                                      1, 1, max_len)

    # policy.save_weights('policy_model_weights.h5')
    # policy.save('policy_model_config.h5')
    # value.save_weights('value_model_weights.h5')
    # value.save('value_model_config.h5')
    # discrim.save_weights('discrim_model_weights.h5')
    # discrim.save('discrim_model_config.h5')


policy.save("./draft/policy_trim_3_embed.h5")
discrim.save("./draft/discrim_trim_3_embed.h5")
value.save("./drgaft/value_trim_3_embed.h5")

learner_observations, learner_actions, learner_len, learner_rewards =unroll_traj(state_inputs_train, goal_inputs_train,
                                                                                    env, policy,
                                                                                    1, 1, max_len)

print(learner_observations)
print(learner_actions)
print(learner_len)
print(learner_rewards)
