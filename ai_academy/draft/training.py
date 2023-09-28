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

    return_values = calculate_return(value_model, learner_obs, learner_act, learner_len)

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
    next_state[:, :-1] = ???

    action_prob = policy_net(data.append(learner_act))
    all_actions = list(itertools.product([0, 1], repeat=4)) # 4 is n_actions
    next_values = [value_net(data.append(action)) for action in all_actions]
    
    expected_return = tf.reduce_sum(next_values * action_prob, axis = 1) + rewards

    return expected_return

def get_reward(model, data):
    prob = model(data)
    return - tf.math.log(tf.clip_by_value(prob, 1e-10, 1))