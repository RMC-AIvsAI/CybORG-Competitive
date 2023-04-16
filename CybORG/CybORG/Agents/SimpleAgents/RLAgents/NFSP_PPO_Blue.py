from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import (
    Monitor,
    Analyse,
    Remove,
    Restore,
    PrivilegeEscalate,
    ExploitRemoteService,
    DiscoverRemoteSystems,
    Impact,
    DiscoverNetworkServices,
    Sleep,
)
import pandas as pd
import numpy as np
from itertools import product
import random
from ast import literal_eval as make_tuple

from collections import deque

import tensorflow as tf

# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Softmax

# tf.keras.backend.set_floatx("float64")

import ray
from ray import tune

# import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env


import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo.ppo_tf_policy import get_ppo_tf_policy
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from gym.spaces import Discrete, MultiDiscrete, MultiBinary, Space, Box
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder

from scipy.special import softmax

# GAMMA = 1  # discount factor, default was 0.99
# UPDATE_INTERVAL = 5
# AVG_LR = 0.0005
# ACTOR_LR = 0.00025  # default was 0.0005
# CRITIC_LR = 0.001  # default was 0.001
# CLIP_RATIO = 0.1
# LAMBDA = 1.0  # default was 0.95
# EPOCHS = 3


# class Critic:
#     def __init__(self, obs_window):
#         self.obs_window = obs_window
#         self.model = self.create_model()
#         self.opt = tf.keras.optimizers.Adam(CRITIC_LR)

#     def create_model(self):
#         return tf.keras.Sequential(
#             [
#                 Input((self.obs_window,)),
#                 Dense(32, activation="relu"),
#                 # Dense(16, activation="relu"),
#                 Dense(16, activation="relu"),
#                 Dense(1, activation="linear"),
#             ]
#         )

#     def compute_loss(self, v_pred, next_value):
#         mse = tf.keras.losses.MeanSquaredError()
#         return mse(next_value, v_pred)

#     def train(self, state, next_value):
#         with tf.GradientTape() as tape:
#             v_pred = self.model(state, training=True)
#             assert v_pred.shape == next_value.shape
#             loss = self.compute_loss(v_pred, tf.stop_gradient(next_value))
#         grads = tape.gradient(loss, self.model.trainable_variables)
#         self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
#         return loss


# class Actor:
#     def __init__(self, obs_window, action_space_list):
#         self.action_space_list = action_space_list

#         self.state_dim = obs_window
#         self.action_dim = len(action_space_list)

#         self.model = self.create_model(obs_window)
#         self.opt = tf.keras.optimizers.Adam(ACTOR_LR)

#     def create_model(self, obs_window):
#         return tf.keras.Sequential(
#             [
#                 Input((obs_window,)),
#                 Dense(32, activation="relu"),  # was 32
#                 Dense(16, activation="relu"),
#                 Dense(len(self.action_space_list), activation="softmax"),
#             ]
#         )

#     def compute_loss(self, old_policy, new_policy, action, delta):
#         delta = tf.stop_gradient(delta)
#         old_log_p = tf.math.log(tf.reduce_sum(old_policy * action))
#         old_log_p = tf.stop_gradient(old_log_p)
#         log_p = tf.math.log(tf.reduce_sum(new_policy * action))
#         ratio = tf.math.exp(log_p - old_log_p)
#         clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
#         surrogate = -tf.minimum(ratio * delta, clipped_ratio * delta)
#         return tf.reduce_mean(surrogate)

#     def train(self, old_policy, state, action, delta):
#         action_vector = []
#         for (
#             item
#         ) in (
#             self.action_space_list
#         ):  # encode the action into a vector across the action space, where only the true action is 1
#             if item == action:
#                 action_vector.append(1)
#             else:
#                 action_vector.append(0)

#         with tf.GradientTape() as tape:
#             logit = self.model(state, training=True)
#             loss = self.compute_loss(old_policy, logit, action_vector, delta)
#         grad = tape.gradient(loss, self.model.trainable_variables)
#         self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
#         return loss


# obs_len should be the size of the observation window, action_num should be the size of the action space
class AveragePolicy:
    def __init__(self, obs_window, action_space_list):

        self.action_space_list = action_space_list
        self.model = self.create_model(obs_window)
        self.opt = tf.keras.optimizers.Adam(AVG_LR)

        self.model.compile(optimizer=self.opt, loss="categorical_crossentropy")

    def create_model(self, obs_window):
        return tf.keras.Sequential(
            [
                Input((obs_window,)),
                Dense(32, activation="relu"),  # was 32
                Dense(16, activation="relu"),
                Dense(len(self.action_space_list), activation="softmax"),
            ]
        )

    def act(self, observation):
        policy = self.model.predict([observation.tolist()], verbose=0)[
            0
        ]  # returns a vector of action probabilities
        select_action = random.uniform(0, 1)
        find_action = 0
        # action = ["Monitor"]
        for index in range(
            len(policy)
        ):  # find which index in the vector was randomly selected
            find_action += policy[index]
            if find_action >= select_action:
                action = self.action_space_list[
                    index
                ]  # decode which action is represented by this index
                break
        return action

    def train(self, observation, action):
        label = []
        for (
            item
        ) in (
            self.action_space_list
        ):  # encode the action into a vector across the action space, where only the true action is 1
            if item == action:
                label.append(1)
            else:
                label.append(0)

        self.model.fit(x=[observation.tolist()], y=[label], verbose=0)


class Reservoir:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, observation, action):
        # state = np.expand_dims(state, 0)
        self.buffer.append((observation, action))

    def get_batch(self, batch_size):
        indices = np.random.randint(low=0, high=len(self.buffer), size=batch_size)
        batch = []
        for i in range(batch_size):
            batch.append(self.buffer[indices[i]])
        return batch


# class PPO:
#     def __init__( *** )

#     def loss(self):


class Blue(BaseAgent):
    def __init__(self):
        # self.host_list = []
        self.observation = None
        self.action = None
        self.action_prob = None
        self.action_logp = None
        self.action_dist_inputs = None
        self.vf_preds = None

        self.last_observation = None
        self.last_action = None
        self.last_action_prob = None
        self.last_action_logp = None
        self.last_action_dist_inputs = None
        self.last_vf_preds = None

        self.reward = 0
        self.last_reward = 0

        self.timesteps_per = 10

        lone_actions = ["Monitor"]  # actions with no parameters
        host_actions = (
            "Analyse",
            "Remove",
            "Restore",
        )  # actions with a hostname parameter
        hostnames = (
            "Enterprise0",
            "Enterprise1",
            "Enterprise2",
            "Op_Host0",
            "Op_Host1",
            "Op_Host2",
            "Op_Server0",
            "User0",
            "User1",
            "User2",
            "User3",
            "User4",
        )
        self.action_space_list = lone_actions + list(product(host_actions, hostnames))
        # self.obs_window = 76  # length of observation vector (65) plus turn vector (11) (10 turns plus DONE flag)

        # self.actor = Actor(self.obs_window, self.action_space_list)
        # self.critic = Critic(self.obs_window)

        # config = ppo.DEFAULT_CONFIG.copy()
        # # config = {}
        # config["postprocess_inputs"] = True
        # config["lr"] = 1e-04

        # # config["model"]["vf_share_layers"] = True
        # # config["model"]["fcnet_hiddens"] = [512, 512]
        # # config["model"]["fcnet_activation"] = "relu"
        # config["model"] = {
        #     "fcnet_hiddens": [64, 64],
        #     "fcnet_activation": "relu",
        #     "use_lstm": False,
        #     "max_seq_len": 10,
        #     # "vf_share_layers": False,
        # }

        # config["num_workers"] = 1
        # config["horizon"] = 10  # Max number of timesteps per episode
        # config["train_batch_size"] = 64
        # config["sgd_minibatch_size"] = 16
        # config["rollout_fragment_length"] = 10
        # config["batch_mode"] = "complete_episodes"
        # config["lambda"] = 0.95
        # config["kl_coeff"] = 0.5
        # config["clip_rewards"] = True
        # config["clip_param"] = 0.01  # epsilon
        # config["vf_clip_param"] = 10.0  # was 10
        # config["entropy_coeff"] = 0.01  # beta
        # config["vf_share_layers"] = True
        # config["actions_in_input_normalized"] = True

        # config["exploration_config"] = {"type": "Random"}

        config = {}
        config["model"] = {
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
            "vf_share_layers": False,
            "max_seq_len": 10,
        }
        config["num_workers"] = 3

        config["train_batch_size"] = 4000
        config["sgd_minibatch_size"] = 128
        config["rollout_fragment_length"] = 200

        config["lr_schedule"] = None
        config["use_critic"] = True
        config["use_gae"] = True
        config["lambda"] = 0.95
        config["kl_coeff"] = 0.2
        # config['num_sgd_iter'] = 30
        config["shuffle_sequences"] = True
        config["vf_loss_coeff"] = 1.0
        config["entropy_coeff"] = 0.0
        config["entropy_coeff_schedule"] = None
        config["clip_param"] = 0.3
        config["vf_clip_param"] = 10.0
        config["grad_clip"] = None
        config["kl_target"] = 0.01
        config["lr"] = 1e-3
        config["vf_share_layers"] = False
        config["batch_mode"] = "truncate_episodes"

        obs_space = MultiDiscrete(np.array([3, 4] * len(hostnames)))
        obs_space = MultiBinary(
            5 * len(hostnames) + self.timesteps_per + 1
        )  # length of observation vector plus turn vector
        # ac_space = Box(low=np.zeros(37), high=np.zeros(37) + 1)
        ac_space = Discrete(37)
        config["observation_space"] = obs_space
        config["action_space"] = ac_space

        self.best = get_ppo_tf_policy("BlueBestResponse", DynamicTFPolicyV2)(
            obs_space=obs_space, action_space=ac_space, config=config
        )

        # self.checkpoint = None
        self.batch_builder = SampleBatchBuilder()

        # capacity from original FSP implementation from deepmind (as poker hands)
        self.sl_buffer = Reservoir(capacity=40000)
        # self.average = AveragePolicy(self.obs_window, self.action_space_list)

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def save_agent(self, name):
        self.actor.model.save_weights(f"./trained_agents/{name}/blue_actor_{name}")
        self.critic.model.save_weights(f"./trained_agents/{name}/blue_critic_{name}")
        self.average.model.save_weights(f"./trained_agents/{name}/blue_average_{name}")

    def load_agent(self, name):
        self.actor.model.load_weights(f"./trained_agents/{name}/blue_actor_{name}")
        self.critic.model.load_weights(f"./trained_agents/{name}/blue_critic_{name}")
        self.average.model.load_weights(f"./trained_agents/{name}/blue_average_{name}")

    def end_episode(self):

        self.observation = None
        self.action = None
        self.action_prob = None
        self.action_logp = None
        self.action_dist_inputs = None
        self.vf_preds = None

        self.last_observation = None
        self.last_action = None
        self.last_action_prob = None
        self.last_action_logp = None
        self.last_action_dist_inputs = None
        self.last_vf_preds = None

        self.reward = None
        self.last_reward = None

    def set_initial_values(self, action_space, observation):
        pass

    def _resolve_action(self, action_index, session):
        action = self.action_space_list[action_index]
        if action[0] == "Analyse":
            return Analyse(hostname=action[1], agent="Blue", session=session)
        elif action[0] == "Remove":
            return Remove(hostname=action[1], agent="Blue", session=session)
        elif action[0] == "Restore":
            return Restore(hostname=action[1], agent="Blue", session=session)
        else:
            return Monitor(agent="Blue", session=session)

    def get_best_action(self, observation, action_space, turn, verbose=False):

        compute_action = self.best.compute_single_action(observation)

        if verbose:
            print(compute_action)
            print(softmax(compute_action[2]["action_dist_inputs"]))

        # assume a single session in the action space
        session = list(action_space["session"].keys())[0]

        # # get stochastic strategy for this observation
        # probs = self.actor.model.predict(
        #     np.reshape(observation, [1, self.obs_window]), verbose=0
        # )
        # # select an action from the strategy
        # action = np.random.choice(self.action_space_list, p=probs[0])

        # self.last_policy = probs
        self.last_action = self.action
        self.last_action_prob = self.action_prob
        self.last_action_logp = self.action_logp
        self.last_action_dist_inputs = self.action_dist_inputs
        self.last_vf_preds = self.vf_preds

        self.action = compute_action[0]
        self.action_prob = compute_action[2]["action_prob"]
        self.action_logp = compute_action[2]["action_logp"]
        # self.action_dist_inputs = softmax(action[2]["action_dist_inputs"])
        self.action_dist_inputs = compute_action[2]["action_dist_inputs"]
        self.vf_preds = compute_action[2]["vf_preds"]

        return self._resolve_action(compute_action[0], session)

    # def get_best_action(self, observation, action_space, turn, verbose=False):

    #     action = self.best.compute_single_action(
    #         obs=observation,
    #         state=[],
    #         # prev_action=self.action,
    #         # prev_reward=self.reward,
    #         timestep=turn,
    #         full_fetch=True,
    #         explore=True,
    #     )
    #     if verbose:
    #         print(action)
    #         print(softmax(action[2]["action_dist_inputs"]))

    #     # assume a single session in the action space
    #     session = list(action_space["session"].keys())[0]

    #     # # get stochastic strategy for this observation
    #     # probs = self.actor.model.predict(
    #     #     np.reshape(observation, [1, self.obs_window]), verbose=0
    #     # )
    #     # # select an action from the strategy
    #     # action = np.random.choice(self.action_space_list, p=probs[0])

    #     # self.last_policy = probs
    #     self.last_action = self.action
    #     self.last_action_prob = self.action_prob
    #     self.last_action_logp = self.action_logp
    #     self.last_action_dist_inputs = self.action_dist_inputs
    #     self.last_vf_preds = self.vf_preds

    #     self.action = action[0]
    #     self.action_prob = action[2]["action_prob"]
    #     self.action_logp = action[2]["action_logp"]
    #     # self.action_dist_inputs = softmax(action[2]["action_dist_inputs"])
    #     self.action_dist_inputs = action[2]["action_dist_inputs"]
    #     self.vf_preds = action[2]["vf_preds"]

    #     return self._resolve_action(action[0], session)

    def get_average_action(self, observation, action_space, turn):

        turn_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # for a 10 turn game
        turn_vector[turn - 1] = 1
        observation = np.concatenate((observation, turn_vector))

        # assume a single session in the action space
        session = list(action_space["session"].keys())[0]

        action = self.average.act(observation)
        return self._resolve_action(action, session)

    def train(self, reward, observation, new_turn):

        turn_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # for a 10 turn game
        turn_vector[new_turn - 1] = 1
        observation = np.concatenate((observation, turn_vector))

        last_obs_value = self.critic.model.predict(
            np.reshape(self.last_observation, [1, self.obs_window]), verbose=0
        )
        next_obs_value = self.critic.model.predict(
            np.reshape(observation, [1, self.obs_window]), verbose=0
        )

        true_value = reward + GAMMA * next_obs_value
        delta = true_value - last_obs_value

        actor_loss = self.actor.train(
            self.last_policy,
            np.reshape(self.last_observation, [1, self.obs_window]),
            self.last_action,
            delta,
        )
        critic_loss = self.critic.train(
            np.reshape(self.last_observation, [1, self.obs_window]), true_value
        )

        # Update SL Buffer
        self.sl_buffer.push(self.last_observation, self.last_action)

        # Train Average Policy
        sample = self.sl_buffer.get_batch(1)[0]
        average_loss = self.average.train(observation=sample[0], action=sample[1])
