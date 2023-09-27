import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences

def Policy_net(
        state_seq,
        seq_len,
        n_actions: int, 
        n_features: int,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64
    ):


    ## StateSeqEmb class in the original code 
    state_input = layers.Input(shape=(n_features,))
    goal_input = layers.Input(shape=(n_features,))
    state_seq_input = layers.Input(shape=(seq_len,))
    embed = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)(state_seq_input)
    padded_embed = pad_sequences(embed, padding='post')
    x_rnn = layers.StackedRNNCells(
        [layers.GRU(hidden_dim),
        layers.GRU(hidden_dim),
        layers.GRU(hidden_dim)])(padded_embed)

    x = layers.Concatenate(axis=1)([x_rnn, goal_input, state_input])

    ## Vanilla Policy class in the original code 
    x = layers.Dense(hidden_dim, activation='relu')(x_rnn)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(n_actions, activation='linear')(x)

    last_states = state_seq[tf.range(state_seq.shape[0]), seq_len-1]
    action_domain = tf.zeros((n_features, n_actions)) # should put to 1 for terminal state

    output = tf.where(tf.cast(1-action_domain[last_states], tf.bool), 1e-32, x)
    # output = layers.Dense(n_actions, activation='relu')(x)
    prob = layers.Softmax(axis=1)(output)
    action_dist = layers.DistributionLambda(
        lambda p: tfp.distributions.Categorical(p))(prob)
    model = Model([state_input, goal_input, state_seq_input], action_dist)

    model.compile(optimizer=Adam(learning_rate=0.01), loss='loss_policy')

    return model

def policy_loss():
    pass