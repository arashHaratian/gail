import tensorflow as tf
import math
import random
import numpy as np

def shorten_traj(sampled_obs, sampled_len, shorten_traj_len):
    ## TO cut the first shorten_traj_len steps ##
    # long_traj_idx = np.where(sampled_len > shorten_traj_len)[0]
    # short_traj_idx = np.where(sampled_len <= shorten_traj_len)[0]

    # new_sampled_obs = np.zeros_like(sampled_obs)
    # for idx in long_traj_idx:
    #     new_sampled_obs[idx, : - shorten_traj_len, :] = sampled_obs[idx, shorten_traj_len : , :]


    # new_sampled_obs[short_traj_idx] = sampled_obs[short_traj_idx]
    # sampled_len[long_traj_idx] -= shorten_traj_len

    # return new_sampled_obs, sampled_len

    ## TO keep the last shorten_traj_len steps ##

    long_traj_idx = np.where(sampled_len > shorten_traj_len)[0]
    short_traj_idx = np.where(sampled_len <= shorten_traj_len)[0]
    new_sampled_obs = np.zeros_like(sampled_obs)
    for idx in long_traj_idx:
        new_sampled_obs[idx, : shorten_traj_len, :] = sampled_obs[idx, sampled_len[idx] - shorten_traj_len : sampled_len[idx], :]

    new_sampled_obs[short_traj_idx] = sampled_obs[short_traj_idx]
    sampled_len[long_traj_idx] = shorten_traj_len

    return new_sampled_obs, sampled_len



def sample_batch(batch, observations, actions, length, start, goal, sort_by_len =True, shorten_traj_len = 0):
    
    data_size = observations.shape[0] ## number of trajs
    idx = list(range(data_size))
    random.shuffle(idx)
    
    for i in range(math.ceil(data_size/batch)): 
        batch_idx = idx[(i*batch):((i+1)*batch)]
        sampled_obs = observations[batch_idx]
        sampled_act =  actions[batch_idx]
        sampled_len = length[batch_idx]
        sampled_start = start[batch_idx]
        sampled_goal = goal[batch_idx]

        if shorten_traj_len:
            sampled_obs, sampled_len = shorten_traj(sampled_obs, sampled_len, shorten_traj_len)
        
        ## In the original code, they sort the input data for both expert and learner!
        if sort_by_len:
            sorted_idx = tf.argsort(sampled_len, direction="DESCENDING")
            sampled_obs = sampled_obs[sorted_idx]
            sampled_act = sampled_act[sorted_idx]
            sampled_len = sampled_len[sorted_idx]
            sampled_start = sampled_start[sorted_idx]
            sampled_goal = sampled_goal[sorted_idx]

        yield sampled_obs, sampled_act, sampled_len, sampled_start, sampled_goal



def train(
        policy_model, value_model, discrim_model,
        env,
        learner_obs, learner_act, learner_len,
        expert_obs, expert_act, expert_len,
        start_state, goal_state,
        num_discrim_update = 2, num_gen_update = 6, batch = 2048):
    

    for _ in range(num_discrim_update):
        learner_loader = sample_batch(batch, learner_obs, learner_act, learner_len, start_state, goal_state, 3)
        expert_loader = sample_batch(batch, expert_obs, expert_act, expert_len, start_state, goal_state, 3)
        
        for (batch_learner_obs, batch_learner_act, _, batch_learner_start_state, batch_learner_goal_state) ,\
        (batch_expert_obs, batch_expert_act, _, batch_expert_start_state, batch_expert_goal_state) in zip(learner_loader, expert_loader):
            discrim_model = train_discrim_step(discrim_model,
                            batch_learner_obs, batch_learner_act,
                            batch_expert_obs, batch_expert_act,
                            batch_learner_start_state, batch_learner_goal_state,
                            batch_expert_start_state, batch_expert_goal_state)



    for _ in range(num_gen_update):
        learner_loader = sample_batch(2048, learner_obs, learner_act, learner_len, start_state, goal_state)
        
        for batch_learner_obs, batch_learner_act, batch_learner_len, batch_learner_start_state, batch_learner_goal_state in learner_loader:
            policy_model, value_model = train_policy_and_value_step(policy_model, value_model, discrim_model, env,
                                        batch_learner_obs, batch_learner_act, batch_learner_len,
                                        batch_learner_start_state, batch_learner_goal_state)
            
    return policy_model, value_model, discrim_model



def train_discrim_step(
    discrim_model,
    learner_obs, learner_act,
    expert_obs, expert_act,
    learner_start_state, learner_goal_state,
    expert_start_state, expert_goal_state):

    with tf.GradientTape() as tape:
        
        learner_target = discrim_model([learner_start_state, learner_goal_state, learner_obs, learner_act])
        expert_target = discrim_model([expert_start_state, expert_goal_state, expert_obs, expert_act])

        real_loss = discrim_model.loss(tf.zeros_like(expert_target), expert_target)
        fake_loss = discrim_model.loss(tf.ones_like(learner_target), learner_target)
        total_loss = real_loss + fake_loss
        
    gradients = tape.gradient(total_loss, discrim_model.trainable_weights)
    discrim_model.optimizer.apply_gradients(zip(gradients, discrim_model.trainable_weights))
    
    # print(f"expert acc : {(expert_target.numpy()<0.5).mean()}  ; learner acc : {(learner_target.numpy()>0.5).mean()}")
    # print(f"discrim loss : {total_loss}")
    
    return discrim_model



def train_policy_and_value_step(
    policy_model, value_model,discrim_model,
    env,
    learner_obs, learner_act, learner_len,
    start_state, goal_state,
    c_1 = 1,
    c_2 = 0.01):

    ## TODO: learner_act = learner_act.squeeze() #CH
    return_values = calculate_return(policy_model, value_model, discrim_model, env, learner_obs, learner_act, learner_len, start_state, goal_state)

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        act_prob = policy_model([start_state, goal_state, learner_obs])
        policy_loss = tf.reduce_mean(act_prob.log_prob(learner_act.squeeze()) * return_values) # maximizing

        val_pred = value_model([start_state, goal_state, learner_obs, learner_act])
        loss_value = value_model.loss(val_pred, tf.expand_dims(return_values, 1)) # minimizing
        
        entropy = tf.reduce_mean(act_prob.entropy()) # maximizing

        loss = - (policy_loss - c_1 * loss_value + c_2 * entropy)
        

    if tf.math.is_inf(policy_loss):
        raise Exception

    gradients = tape1.gradient(loss, policy_model.trainable_weights)
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_weights))
    
    gradients = tape2.gradient(loss, value_model.trainable_weights)
    value_model.optimizer.apply_gradients(zip(gradients, value_model.trainable_weights))

    # print(policy_loss, loss_value, entropy)
    # print(f"policy loss : {loss}")

    return policy_model, value_model



def calculate_return(
    policy_net, value_net, discrim_model,
    env,
    learner_obs, learner_act, learner_len,
    start_state, goal_state, gamma = 0.95): #CH 0.99

    batch_size = learner_obs.shape[0]
    discrim_rewards = get_reward(discrim_model, [start_state, goal_state, learner_obs, learner_act])
    current_state = learner_obs[np.arange(batch_size), learner_len-1, :]
    new_states, _, is_terminal_new_states = env.step_vectorized(current_state, tf.convert_to_tensor(learner_act.squeeze(1))) ## Change actions from (batch, 1) to (batch, )

    new_learner_obs = np.zeros((batch_size, learner_obs.shape[1] + 1, 3))
    new_learner_obs[:, :learner_obs.shape[1], :] = learner_obs
    new_learner_obs[np.arange(batch_size), learner_len, :] = new_states

    action_prob = policy_net([start_state, goal_state, new_learner_obs])

    all_actions = np.arange(6) # 6 is n_actions
    next_values = [value_net([start_state, goal_state, new_learner_obs, np.repeat(np.expand_dims([action], 0), batch_size, 0)]) for action in all_actions]
    next_values = np.concatenate(next_values, axis = 1)
    next_values[is_terminal_new_states] = 0 ## remove the estimated value of the terminal nodes

    # print(tf.reduce_mean(rewards).numpy(),"  ", tf.reduce_mean(discrim_rewards).numpy(), "  ",  tf.reduce_mean(rewards + discrim_rewards).numpy())
    # if(np.any(np.all(goal_state[0, :] == current_state, axis = 1))):
    #     raise Exception
    # if(np.any(is_terminal_new_states)):
    #     print("here")
    expected_return = gamma * tf.reduce_sum(next_values * action_prob.probs, axis = 1) + tf.squeeze(discrim_rewards)

    return expected_return



def get_reward(model, data):
    prob = model(data)

    return - tf.math.log(tf.clip_by_value(prob, 1e-10, 1)) 
    # return - tf.math.log(tf.clip_by_value(prob, 1e-10, 1))/25 #CH



def unroll_traj(
        start_obs, goal_state,
        env, policy_net,
        batch_size, num_trajs, max_len
        ):
    
    learner_obs = -1 * np.ones((num_trajs, max_len + 1, 3))
    learner_actions = -1 * np.ones((num_trajs, max_len))
    learner_len = np.zeros((num_trajs, ), dtype=np.int32)
    learner_reward = np.zeros((num_trajs, max_len))

    out_max_length = 0
    processed = 0

    for i in range(int(np.ceil(num_trajs / batch_size))):

        if num_trajs - processed >= batch_size:
            batch_obs, batch_act, batch_len, batch_reward = unroll_batch(start_obs[(i*batch_size):((i+1)*batch_size), :], goal_state[(i*batch_size):((i+1)*batch_size), :],
                                                                        env, policy_net, max_len)
            
            batch_max_length = batch_obs.shape[1]

            
            learner_obs[(i*batch_size):((i+1)*batch_size), :batch_max_length, :] = batch_obs
            learner_actions[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_act
            learner_len[(i*batch_size):((i+1)*batch_size)] = batch_len
            learner_reward[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_reward
            processed += batch_obs.shape[0]
        else:
            batch_obs, batch_act, batch_len, batch_reward = unroll_batch(start_obs[(i*batch_size):, :], goal_state[(i*batch_size):((i+1)*batch_size), :], 
                                                                        env, policy_net, max_len)  
                                                                                  
            batch_max_length = batch_obs.shape[1]


            learner_obs[(i*batch_size):, :batch_max_length, :] = batch_obs
            learner_actions[(i*batch_size):, :(batch_max_length-1)] = batch_act
            learner_len[(i*batch_size):] = batch_len
            learner_reward[(i*batch_size):, :(batch_max_length-1)] = batch_reward
        
        out_max_length = max(out_max_length, batch_max_length)

    learner_obs = learner_obs[:, :out_max_length, :]
    learner_actions = learner_actions[:, :(out_max_length-1)]
    learner_reward = learner_reward[:, :(out_max_length-1)]

    return learner_obs, learner_actions, learner_len, learner_reward



def unroll_batch(
        start_obs, goal_state,
        env, policy_net, max_len
        ):
    
    batch_size = start_obs.shape[0]
    obs = np.expand_dims(start_obs, 1) ## Making a (batch, 3) tensor to (batch, len_of_seq, 3) where len_of_seq is 1 now
    obs_len = np.ones((batch_size), dtype=np.int32) ## Size of len_of_seq for each batch
    actions = np.zeros((batch_size, 1))
    rewards = np.zeros((batch_size, 1))
    last_actions = -1 * np.ones((batch_size))


    # CH
    # obs = np.zeros((batch_size, max_len, 3))
    # obs[np.arange(batch_size), 0, :] = start_obs
    # obs_len = np.ones((batch_size), dtype=np.int32) ## Size of len_of_seq for each batch
    # actions = np.zeros((batch_size, max_len))
    # rewards = np.zeros((batch_size, max_len))

    done_mask = np.zeros((batch_size),dtype=bool)


    for i in range(max_len):
        

        ## Selects samples that are not done yet
        notdone_obs = obs[~done_mask, :, :]
        if notdone_obs.shape[0] == 0:
            break

        ## select the last in each batch (len_of_seq = to the last)
        action_dist = policy_net([notdone_obs[:, 0, :], goal_state[~done_mask, :], notdone_obs])

        action = action_dist.sample()

        # TL Logic
        if i == 0:
            last_actions_np = action.numpy().copy()

        all_values_equal_to_minus_one = tf.reduce_all(tf.equal(last_actions, -1))
        if all_values_equal_to_minus_one == False:
            action_np = action.numpy()
            aux_last_actions_np = tf.boolean_mask(last_actions, ~done_mask).numpy()
            pairs_to_swap = [(0, 1), (2, 3), (4, 5)]

            for pair in pairs_to_swap:
                mask = ((action_np == pair[0]) & (aux_last_actions_np == pair[1])) | ((action_np == pair[1]) & (aux_last_actions_np == pair[0]))
                action_np[mask], aux_last_actions_np[mask] = aux_last_actions_np[mask], action_np[mask]

            action = tf.constant(action_np)
            last_actions = tf.constant(last_actions_np)
        
            false_indices = np.where(~done_mask)
            last_actions_np[false_indices] = action_np

        last_actions = tf.constant(last_actions_np)

        if tf.reduce_max(action) > 5:
            raise Exception
        if tf.reduce_min(action) < 0:
            raise Exception
        
        new_state, reward, done = env.step_vectorized(notdone_obs[:, -1, :], action)
        new_column = np.zeros((batch_size, 1, 3)) # since zeros are masked
        new_column[~done_mask, 0, :] = new_state
        obs = np.concatenate([obs, new_column], 1) ## Concat should be in second dim (len_of_seq)
        # obs[~done_mask, i, :] = new_state #CH
        
        new_column = np.zeros((batch_size, 1))
        new_column[~done_mask, 0] = action
        actions = np.concatenate([actions, new_column], 1) ## Concat should be in second dim (len_of_seq)
        # actions[~done_mask, i] = action #CH

        new_column[~done_mask] = reward
        rewards = np.concatenate([rewards, new_column], 1) ## Concat should be in second dim (len_of_seq)
        # rewards[~done_mask, i] = reward #CH

        
        done_mask[~done_mask] = done 
        obs_len += 1-done_mask
        
        ##### DEBUGGING ###########
        # print(obs.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # if tf.reduce_any(done):
        #     newdone_mask = tf.zeros((batch_size), tf.bool).numpy()
        #     newdone_mask[~done_mask] = done 
        #     iii = tf.where(newdone_mask)
        #     print(iii)
        #     print(done_mask[iii])
        #     done_mask[~done_mask] = done 
        #     obs_len += 1-done_mask
        #     print(done_mask[iii])
        # else:
        #     done_mask[~done_mask] = done 
        #     obs_len += 1-done_mask
        
    actions = actions[:, 1:]
    rewards = rewards[:, 1:]

    # CH
    # actions = actions.squeeze(-1)
    # rewards = rewards.squeeze(-1)
    
    return obs, actions, obs_len, rewards
