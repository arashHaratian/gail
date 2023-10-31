# from draft import  discrim_net, policy_net, value_net
import tensorflow as tf
import itertools


def train(
        policy_model, value_model, discrim_model,
        learner_obs, learner_act, learner_len,
        expert_obs, expert_act, expert_len,
        current_state, goal_state,
        num_discrim_update = 2, num_gen_update = 6):
    

    for _ in range(num_discrim_update):
        
        ## In the original code, they sort the input data for both expert and learner!
        train_discrim_step(discrim_model,
                           learner_obs, learner_act, learner_len,
                           expert_obs, expert_act, expert_len,
                           current_state, goal_state)

    for _ in range(num_gen_update):
        train_policy_and_value_step(policy_model, value_model,
                                    learner_obs, learner_act, learner_len,
                                    current_state, goal_state)


def train_discrim_step(
    discrim_model,
    learner_obs, learner_act, learner_len,
    expert_obs, expert_act, expert_len,
    current_state, goal_state):
    
    ## ================= solution 1 ==================================
    
    with tf.GradientTape() as tape:
        
        learner_target = discrim_model([current_state, goal_state, learner_obs, learner_act])
        expert_target = discrim_model([current_state, goal_state, expert_obs, expert_act])

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) ## TODO if from_logit should be there or no!! if yes add it to the nework optim too
        real_loss = cross_entropy(tf.ones_like(expert_target), expert_target)
        fake_loss = cross_entropy(tf.zeros_like(learner_target), learner_target)
        total_loss = real_loss + fake_loss
        
    gradients = tape.gradient(total_loss, discrim_model.trainable_weights)
    discrim_model.optimizer.apply_gradients(zip(gradients, discrim_model.trainable_weights))

            

    ## ================= solution 2 ==================================
    input_data = tf.concat([[current_state, goal_state, learner_obs, learner_act],
                  [current_state, goal_state, expert_obs, expert_act]], axis = 0) 
    target_data = tf.concat([tf.zeros(learner_len,), tf.ones(expert_len,)], axis = 0)
    discrim_model.fit(input_data, target_data)
    
    return



def train_policy_and_value_step(
    policy_model,
    value_model,
    learner_obs, learner_act, learner_len,
    current_state, goal_state,
    c_1 = 1,
    c_2 = 0.01):

    return_values = calculate_return(value_model, learner_obs, learner_act, learner_len) ## TODO maybe inside the tapes?

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        act_prob = policy_model([current_state, goal_state, learner_obs, learner_act])
        policy_loss = (act_prob.log_prob(act_prob) * return_values).mean()

        val_pred = value_model([current_state, goal_state, learner_obs, learner_act])
        loss_value = value_model.optimizer(val_pred, return_values)
        
        entropy = act_prob.entropy().mean()

        loss = - (policy_loss - c_1 * loss_value + c_2 * entropy)

    gradients = tape1.gradient(loss, policy_model.trainable_weights)
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_weights))
    
    gradients = tape2.gradient(loss, value_model.trainable_weights)
    value_model.optimizer.apply_gradients(zip(gradients, value_model.trainable_weights))
        

    return



def calculate_return(
    policy_net, value_net, discrim_model,
    learner_obs, learner_act, learner_len,
    current_state, goal_state):

    data = [current_state, goal_state, learner_obs]
    rewards = get_reward(discrim_model, data)

    ## TODO Get the ext state
    next_state = tf.ones((learner_obs.shape[0], learner_obs.shape[1] + 1))
    next_state[:, :-1] = "???"

    action_prob = policy_net(data.append(learner_act))
    all_actions = list(itertools.product([0, 1], repeat=4)) # 4 is n_actions
    next_values = [value_net(data.append(action)) for action in all_actions]
    
    expected_return = tf.reduce_sum(next_values * action_prob, axis = 1) + rewards

    return expected_return

def get_reward(model, data):
    prob = model(data)
    return - tf.math.log(tf.clip_by_value(prob, 1e-10, 1))


def unroll_traj(
        start_obs, goal_state,
        env, policy_net,
        batch_size, num_trajs, max_len
        ):
    
    learner_obs = -1 * tf.ones((num_trajs, max_len, 3))
    learner_actions = -1 * tf.ones((num_trajs, ))
    learner_len = tf.zeros((num_trajs, ))
    learner_reward = tf.zeros((num_trajs, max_len))

    out_max_length = 0
    processed = 0

    for i in range(int(num_trajs/ batch_size)):
        # batch_obs, batch_act, batch_len, batch_reward = unroll_batch(start_obs, goal_state, env, policy_net,
        #                                                             batch_size, max_len
        #                                                              )

        if num_trajs - processed > batch_size:
            batch_obs, batch_act, batch_len, batch_reward = unroll_batch(start_obs[(i*batch_size):((i+1)*batch_size), :], goal_state,
                                                                        env, policy_net,
                                                                        batch_size, max_len)
            
            batch_max_length = batch_obs.shape[1]

            
            learner_obs[(i*batch_size):((i+1)*batch_size), :batch_max_length, :] = batch_obs
            learner_actions[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_act
            learner_len[(i*batch_size):((i+1)*batch_size)] = batch_len
            learner_reward[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_reward
            processed += batch_obs.shape[0]
        else:
            batch_obs, batch_act, batch_len, batch_reward = unroll_batch(start_obs[(i*batch_size):, :], goal_state, 
                                                                        env, policy_net,
                                                                        batch_size, max_len)  
                                                                                  
            batch_max_length = batch_obs.shape[1]


            learner_obs[(i*batch_size):((i+1)*batch_size), :batch_max_length, :] = batch_obs[:(num_trajs - processed)]
            learner_actions[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_act[:(num_trajs - processed)]
            learner_len[(i*batch_size):((i+1)*batch_size)] = batch_len[:(num_trajs - processed)]
            learner_reward[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)] = batch_reward[:(num_trajs - processed)]
        
        out_max_length = max(out_max_length, batch_max_length)

    learner_obs = learner_obs[:, :out_max_length, :]
    learner_actions = learner_actions[:, :(out_max_length-1)]
    learner_reward = learner_reward[:, :(out_max_length-1)]

    return learner_obs, learner_actions, learner_len, learner_reward




def unroll_batch(
        start_obs, goal_state,
        env, policy_net,
        batch_size, max_len
        ):
    
    obs = tf.expand_dims(start_obs, 1) ## Making a (batch, 3) tensor to (batch, len_of_seq, 3) where len_of_seq is 1 now
    obs_len = tf.ones((batch_size)) ## Size of len_of_seq for each batch
    actions = tf.zeros((batch_size, 1))
    rewards = tf.zeros((batch_size, 1))

    done_mask = tf.zeros((batch_size), tf.bool)


    for i in range(max_len):
        new_column = tf.zeros((batch_size, 1)) # since zeros are masked
        new_column = new_column.numpy()

        ## Selects samples that are not done yet
        notdone_obs = tf.boolean_mask(obs, ~done_mask, axis=0)
        # notdone_obs_len = obs_len[~done_mask]

        if notdone_obs.shape[0] == 0:
            break

        ## select the last in each batch (len_of_seq = to the last)
        action_dist = policy_net([notdone_obs[:, 0, :], goal_state, notdone_obs]) ##TODO check the start and the goal data

        action = action_dist.sample()
        action = tf.one_hot(action, action_dist.logits.shape[1]) ## action_dist.logits.shape should be (batch_size,n_actions)

        new_state, reward, done = env.step_vectorized(notdone_obs, action) ##TODO

        new_column[~done_mask] = new_state
        obs = tf.concat([obs, new_column], 1) ## Concat should be in second dim (len_of_seq) ##TODO check the dim in case!
        
        
        new_column[~done_mask] = action
        actions = tf.concat([actions, new_column], 1) ## Concat should be in second dim (len_of_seq)


        new_column[~done_mask] = reward
        rewards = tf.concat([rewards, new_column], 1) ## Concat should be in second dim (len_of_seq)

        done_mask = done 
        obs_len += 1-done_mask

    actions = actions[:, 1:]
    rewards = rewards[:, 1:]
    
    return obs, actions, obs_len, rewards
