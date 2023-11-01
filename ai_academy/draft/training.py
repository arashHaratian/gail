# from draft import  discrim_net, policy_net, value_net
import tensorflow as tf
import random

def sample_batch(batch, observations, actions, length, start, goal, sort_by_len =True):
    
    data_size = observations.shape[0] ## number of trajs
    idx = random.choices(range(data_size), k = batch)
    
    sampled_obs = observations[idx]
    sampled_act =  actions[idx]
    sampled_len = length[idx]
    sampled_start = start[idx]
    sampled_goal = goal[idx]
    
    if sort_by_len:
        sorted_idx = tf.argsort(sampled_len, direction="DESCENDING")
        sampled_obs = sampled_obs[sorted_idx]
        sampled_act = sampled_act[sorted_idx]
        sampled_len = sampled_len[sorted_idx]
        sampled_start = sampled_start[sorted_idx]
        sampled_goal = sampled_goal[sorted_idx]

    return sampled_obs, sampled_act, sampled_len, sampled_start, sampled_goal



def train(
        policy_model, value_model, discrim_model,
        env,
        learner_obs, learner_act, learner_len,
        expert_obs, expert_act, expert_len,
        start_state, goal_state,
        num_discrim_update = 2, num_gen_update = 6, batch = 2048):
    

    for _ in range(num_discrim_update):

        learner_obs, learner_act, learner_len, learner_start_state, learner_goal_state = sample_batch(batch,
                                                                                    learner_obs,
                                                                                    learner_act,
                                                                                    learner_len,
                                                                                    start_state, goal_state)
        expert_obs, expert_act, expert_len, expert_start_state, expert_goal_state = sample_batch(batch,
                                                                                    expert_obs,
                                                                                    expert_act,
                                                                                    expert_len,
                                                                                    start_state, goal_state)
            
        
        ## In the original code, they sort the input data for both expert and learner!
        train_discrim_step(discrim_model,
                           learner_obs, learner_act, learner_len,
                           expert_obs, expert_act, expert_len,
                           learner_start_state, learner_goal_state,
                           expert_start_state, expert_goal_state)



    for _ in range(num_gen_update):
        learner_obs, learner_act, learner_len, learner_start_state, learner_goal_state = sample_batch(batch,
                                                                                    learner_obs,
                                                                                    learner_act,
                                                                                    learner_len,
                                                                                    start_state, goal_state)
        expert_obs, expert_act, expert_len, expert_start_state, expert_goal_state = sample_batch(batch,
                                                                                    expert_obs,
                                                                                    expert_act,
                                                                                    expert_len,
                                                                                   start_state, goal_state)
        
        
        train_policy_and_value_step(policy_model, value_model, env,
                                    learner_obs, learner_act, learner_len,
                                    learner_start_state, learner_goal_state,
                                    expert_start_state, expert_goal_state)



def train_discrim_step(
    discrim_model,
    learner_obs, learner_act, learner_len,
    expert_obs, expert_act, expert_len,
    learner_start_state, learner_goal_state,
    expert_start_state, expert_goal_state):
    
    ## ================= solution 1 ==================================
    
    with tf.GradientTape() as tape:
        
        learner_target = discrim_model([learner_start_state, learner_goal_state, learner_obs, learner_act])
        expert_target = discrim_model([expert_start_state, expert_goal_state, expert_obs, expert_act])

        # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = discrim_model.loss(tf.ones_like(expert_target), expert_target)
        fake_loss = discrim_model.loss(tf.zeros_like(learner_target), learner_target)
        total_loss = real_loss + fake_loss
        
    gradients = tape.gradient(total_loss, discrim_model.trainable_weights)
    discrim_model.optimizer.apply_gradients(zip(gradients, discrim_model.trainable_weights))

            

    ## ================= solution 2 ==================================
    # input_data = tf.concat([[learner_start_state, learner_goal_state, learner_obs, learner_act],
    #               [expert_start_state, expert_goal_state, expert_obs, expert_act]], axis = 0) 
    # target_data = tf.concat([tf.zeros(learner_len,), tf.ones(expert_len,)], axis = 0)
    # discrim_model.fit(input_data, target_data)
    
    return



def train_policy_and_value_step(
    policy_model, value_model,
    env,
    learner_obs, learner_act, learner_len,
    start_state, goal_state,
    c_1 = 1,
    c_2 = 0.01):

    return_values = calculate_return(policy_model, value_model, env, learner_obs, learner_act, learner_len, start_state, goal_state) ## TODO maybe inside the tapes?

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        act_prob = policy_model([start_state, goal_state, learner_obs, learner_act])
        policy_loss = (act_prob.log_prob(act_prob) * return_values).mean()

        val_pred = value_model([start_state, goal_state, learner_obs, learner_act])
        loss_value = value_model.optimizer(val_pred, return_values)
        
        entropy = act_prob.entropy().mean()

        loss = - (policy_loss - c_1 * loss_value + c_2 * entropy)

    gradients = tape1.gradient(loss, policy_model.trainable_weights)
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_weights))
    
    gradients = tape2.gradient(loss, value_model.trainable_weights)
    value_model.optimizer.apply_gradients(zip(gradients, value_model.trainable_weights))
        

    return



def calculate_return(
    policy_net, value_net,
    # discrim_model,
    env,
    learner_obs, learner_act, learner_len,
    start_state, goal_state):

    batch_size = learner_obs.shape[0]
    # rewards = get_reward(discrim_model, data)
    current_state = learner_obs[:, learner_len-1,:]
    new_states, rewards, _ = env.step_vectorized(current_state, learner_act)

    ## TODO Get the ext state
    new_learner_obs = tf.zeros((batch_size, learner_obs.shape[1] + 1, 3))
    new_learner_obs[:, :learner_act, :] = learner_obs
    new_learner_obs[:, learner_act, :] = new_states

    action_prob = policy_net([start_state, goal_state, new_learner_obs])

    all_actions = tf.stack([tf.one_hot(i, 6) for i in range(6)]) # 6 is n_actions
    next_values = [value_net([start_state, goal_state, new_learner_obs, tf.repeat(tf.expand_dims(action, 0), batch_size, 0)]) for action in all_actions]
    next_values = tf.concat(next_values, axis = 1)
    expected_return = tf.reduce_sum(next_values * action_prob.probs(), axis = 1) + rewards

    return expected_return



def get_reward(model, data):
    prob = model(data)
    return - tf.math.log(tf.clip_by_value(prob, 1e-10, 1))



def unroll_traj(
        start_obs, goal_state,
        env, policy_net,
        batch_size, num_trajs, max_len
        ):
    
    learner_obs = -1 * tf.ones((num_trajs, max_len + 1, 3)).numpy()
    learner_actions = -1 * tf.ones((num_trajs, max_len)).numpy()
    learner_len = tf.zeros((num_trajs, ), dtype=tf.dtypes.int32).numpy()
    learner_reward = tf.zeros((num_trajs, max_len)).numpy()

    out_max_length = 0
    processed = 0

    for i in range(int(tf.math.ceil(num_trajs / batch_size))):

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
    obs = tf.expand_dims(start_obs, 1) ## Making a (batch, 3) tensor to (batch, len_of_seq, 3) where len_of_seq is 1 now
    obs_len = tf.ones((batch_size), dtype=tf.dtypes.int32) ## Size of len_of_seq for each batch
    actions = tf.zeros((batch_size, 1))
    rewards = tf.zeros((batch_size, 1))

    done_mask = tf.zeros((batch_size), tf.bool).numpy()


    for i in range(max_len):
        

        ## Selects samples that are not done yet
        notdone_obs = tf.boolean_mask(obs, ~done_mask, axis=0)
        # notdone_obs_len = obs_len[~done_mask]
        # print(f"------{notdone_obs.shape[0]} more")
        if notdone_obs.shape[0] == 0:
            break

        ## select the last in each batch (len_of_seq = to the last)
        action_dist = policy_net([notdone_obs[:, 0, :], tf.boolean_mask(goal_state, ~done_mask, axis=0), notdone_obs]) ##TODO check the start and the goal data

        action = action_dist.sample()
        if tf.reduce_max(action) > 5:
            raise Exception
        if tf.reduce_min(action) < 0:
            raise Exception
        
        new_state, reward, done = env.step_vectorized(notdone_obs[:, -1, :], action) ##TODO

        new_column = tf.zeros((batch_size, 1, 3)).numpy() # since zeros are masked
        new_column[~done_mask, 0, :] = new_state
        obs = tf.concat([obs, new_column], 1) ## Concat should be in second dim (len_of_seq) ##TODO check the dim in case!
        
        
        new_column = tf.zeros((batch_size, 1)).numpy()
        new_column[~done_mask, 0] = action
        actions = tf.concat([actions, new_column], 1) ## Concat should be in second dim (len_of_seq)

        new_column = tf.zeros((batch_size, 1)).numpy()
        new_column[~done_mask] = reward
        rewards = tf.concat([rewards, new_column], 1) ## Concat should be in second dim (len_of_seq)

        
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
    
    return obs, actions, obs_len, rewards
