import mss

import gym
import gym_LoL

import tensorflow as tf
import numpy as np
import random
import cv2

import cv2
import numpy as np

import json
import os

import numpy as np

import tensorflow as tf


class DRQNAgent(object):
    """Implements a standard DDDQN agent"""

    def __init__(self,
                 drqn,
                 target_drqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=100,
                 replay_buffer_start_size=50,
                 max_frames=2500,
                 input_depth=1):

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        self.input_depth = input_depth

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames


        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # DQN
        self.drqn = drqn
        self.target_dqn = target_drqn

    def calc_epsilon(self, frame_number, evaluation=False):

        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.drqn.predict(
            state.reshape((-1, self.history_length, self.input_shape[0], self.input_shape[1], self.input_depth)))[0]
        return q_vals.argmax()

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):

        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.drqn.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.drqn.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.drqn.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(
            state.reshape((-1, self.history_length, self.input_shape[0], self.input_shape[1], self.input_depth)))

    def update_target_network(self):

        self.target_dqn.set_weights(self.drqn.get_weights())

    def add_experience(self, action, frame, reward, terminal):

        self.replay_buffer.add_experience(action, frame, reward, terminal)

    def learn(self, batch_size, gamma):

        states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(
            batch_size=self.batch_size)

        # Main DQN estimates best action in new states
        arg_q_max = self.drqn.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma * double_q * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.drqn(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

        model_gradients = tape.gradient(loss, self.drqn.trainable_variables)
        self.drqn.optimizer.apply_gradients(zip(model_gradients, self.drqn.trainable_variables))

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.drqn.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.drqn = tf.keras.models.load_model(folder_name + '/dqn.h5')

        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.drqn.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']
        print('load buffer')
        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta
