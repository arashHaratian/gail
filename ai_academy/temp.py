import tensorflow as tf
from draft.discrim_net import Discrim_net
from draft.policy_net import Policy_net
from draft.value_net import Value_net
from draft.training import *
from maze_env import Maze
from environment import get_obstacles

batch = 2048
n_space = 3
n_actions = 6
max_len = 200
num_trajs = batch


env_dim = tf.constant([40, 40, 6])

n_features = 40 + 1

## Setting up the env
obstacles, obstacles_x, obstacles_y, obstacles_z = get_obstacles()
env = Maze(obstacles)


## TODO randomize the starts

start_state = tf.reshape(env.reset(), (1, -1))
state_inputs_train  = tf.repeat(start_state, batch, 0)

end_state = tf.reshape(env.end_node, (1, -1))
goal_inputs_train  = tf.repeat(end_state, batch, 0)


## TODO use bi_a_star_2.py
y_train_action = tf.random.uniform((batch, n_actions))


discrim = Discrim_net(n_actions, n_features) 
policy = Policy_net(n_actions, n_features)
value = Value_net(n_actions, n_features)

learner_observations, learner_actions, learner_len, learner_rewards =unroll_traj(state_inputs_train, goal_inputs_train,
                                                                                env, policy,
                                                                                batch, num_trajs, max_len)


learner_obs = tf.zeros((learner_len.sum(), learner_len.max(), n_space)) ## zeros since we mask them in RNN ##TODO problem of cordinate zero!
learner_act = tf.zeros((learner_len.sum()))
learner_l = tf.zeros((learner_len.sum()))
cnt = 0

## TODO: needs to be check (just copied and added the last dimension, not sure if it works)
for sample in range(num_trajs):
    for seq_length in range(1, learner_len[sample]+1):
        try:
            learner_obs[cnt, :seq_length, :] = learner_observations[sample, :seq_length, :]
            learner_act[cnt] = int(learner_actions[sample][seq_length-1])
            learner_l[cnt] = seq_length
            cnt += 1
        except:
            # print("break with index error in Learner Trajectory")
            break

expert_observations, expert_actions, expert_len = expert_data

train(policy, value, discrim,
    learner_obs, learner_act, learner_len,
    expert_observations, expert_actions, expert_len,
    state_inputs_train, goal_inputs_train,
    num_discrim_update = 2, num_gen_update = 6)