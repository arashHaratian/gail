import tensorflow as tf
from ai_academy.draft.discrim_net import Discrim_net
from ai_academy.draft.policy_net import Policy_net
from ai_academy.draft.value_net import Value_net
from ai_academy.draft.training import *
batch = 2048
n_space = 3
n_actions = 6
max_len = 100
num_trajs = batch
state_inputs_train = tf.random.uniform((batch, n_space))
goal_inputs_train = tf.random.uniform((batch, n_space))
y_train_action = tf.random.uniform((batch, n_actions))

seq_len = 200
env_dim = tf.constant([40, 40, 6])

## TODO seq_len ## In the original code it is 50 (number of states) 
## should ours be 3d?! ## it is used for defininig the length of embeding
discrim = Discrim_net(seq_len, n_actions, n_space) ## TODO seq_len 
policy = Policy_net(seq_len, n_actions, n_space) ## TODO seq_len
value = Value_net(seq_len, n_actions, n_space) ## TODO seq_len

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