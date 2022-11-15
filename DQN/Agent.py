import mss

import gym
import gym_LoL

import tensorflow as tf
import numpy as np
import random
import cv2

import cv2
import numpy as np

#
# class Agent:
#     def __init__(self, env, exp_buffer):
#         self.state = None
#         self.env = env
#         self.exp_buffer = exp_buffer
#         self._reset()
#
#     def _reset(self):
#         self.state = np.repeat(self.process_frame(self.env.reset()), 1, axis=2)
#         self.total_reward = 0.0
#
#     def process_frame(self, frame, shape=(640, 640)):
#         """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
#         Arguments:
#             frame: The frame to process.  Must have values ranging from 0-255
#         Returns:
#             The processed frame
#         """
#         frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = frame[34:34 + 160, :160]  # crop image
#         frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
#         frame = frame.reshape((*shape, 1))
#
#         return frame
#
#     def play_step(self, net, epsilon=0.0):
#         done_reward = None
#
#         if np.random.random() < epsilon:
#             action = np.random.randint(0, self.env.action_space.n)
#         else:
#             action = self.DQN.predict(self.state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
#             print(action)
#
#         new_state, reward, is_done, playerState = self.env.step(action)
#         new_state = np.repeat(self.process_frame(new_state), 1, axis=2)
#         self.total_reward += reward
#
#         exp = Experience(self.state, action, reward, is_done, new_state)
#         self.exp_buffer.append(exp)
#         self.state = new_state
#         if is_done:
#             done_reward = self.total_reward
#             self._reset()
#         return done_reward,playerState

import json
import os

import numpy as np

import tensorflow as tf


class Agent(object):
    """Implements a standard DDDQN agent"""

    def __init__(self,
                 dqn,
                 target_dqn,
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
                 max_frames=2500):

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length

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

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        # 根据帧数计算epsilon
        #在前50个动作，epsilon是1
        #随后逐渐递减
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        return q_vals.argmax()
    # 没有用处
    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
            stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame
        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, frame, reward, terminal)

    def learn(self, batch_size, gamma):

        states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(
            batch_size=self.batch_size)

        # Main DQN estimates best action in new states
        # 得到最高的Q值的动作
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)
        # Target DQN estimates q-vals for new states
        #得到所有动作的Q值
        future_q_vals = self.target_dqn.predict(new_states)
        #在targetQ中得到和main Q相同动作的Q值
        double_q = future_q_vals[range(batch_size), arg_q_max]


        # Calculate targets (bellman equation)
        target_q = rewards + (gamma * double_q * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions,
                                                            dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)


        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))


        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')
        print("start loading model")
        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        print("model is loaded")

        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer
        print("model is loaded")
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
