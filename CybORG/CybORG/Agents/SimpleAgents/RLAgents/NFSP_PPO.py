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
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Softmax

tf.keras.backend.set_floatx("float64")

GAMMA = 1  # discount factor, default was 0.99
UPDATE_INTERVAL = 5
AVG_LR = 0.01
ACTOR_LR = 0.0005  # default was 0.0005
CRITIC_LR = 0.001  # default was 0.001
CLIP_RATIO = 0.1
LAMBDA = 1.0  # default was 0.95
EPOCHS = 3


class Critic:
    def __init__(self, obs_window):
        self.obs_window = obs_window
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(CRITIC_LR)

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.obs_window,)),
                # Dense(32, activation="relu"),
                # Dense(16, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

    def compute_loss(self, v_pred, next_value):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(next_value, v_pred)

    def train(self, state, next_value):
        with tf.GradientTape() as tape:
            v_pred = self.model(state, training=True)
            assert v_pred.shape == next_value.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(next_value))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Actor:
    def __init__(self, obs_window, action_space_list):
        self.action_space_list = action_space_list

        self.state_dim = obs_window
        self.action_dim = len(action_space_list)

        self.model = self.create_model(obs_window)
        self.opt = tf.keras.optimizers.Adam(ACTOR_LR)

    def create_model(self, obs_window):
        return tf.keras.Sequential(
            [
                Input((obs_window,)),
                Dense(48, activation="relu"),  # was 32
                # Dense(16, activation="relu"),
                Dense(len(self.action_space_list), activation="softmax"),
            ]
        )

    def compute_loss(self, old_policy, new_policy, action, delta):
        delta = tf.stop_gradient(delta)
        old_log_p = tf.math.log(tf.reduce_sum(old_policy * action))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(new_policy * action))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
        surrogate = -tf.minimum(ratio * delta, clipped_ratio * delta)
        return tf.reduce_mean(surrogate)

    def train(self, old_policy, state, action, delta):
        action_vector = []
        for (
            item
        ) in (
            self.action_space_list
        ):  # encode the action into a vector across the action space, where only the true action is 1
            if item == action:
                action_vector.append(1)
            else:
                action_vector.append(0)

        with tf.GradientTape() as tape:
            logit = self.model(state, training=True)
            loss = self.compute_loss(old_policy, logit, action_vector, delta)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss


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
                Dense(48, activation="relu"),  # was 32
                # Dense(16, activation="relu"),
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


class BlueAC(BaseAgent):
    def __init__(self):
        # self.host_list = []
        self.last_observation = None
        self.last_policy = None
        self.last_action = None

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
        self.obs_window = 65  # length of observation vector

        # self.actor = pd.DataFrame(
        #     columns=self.action_space_list
        # )  # rows are states, columns are actions, cells hold probabilities for each state-action
        # self.critic = pd.DataFrame(
        #     columns=self.action_space_list
        # )  # rows are states, columns are actions, cells hold the state-action value

        self.actor = Actor(self.obs_window, self.action_space_list)
        self.critic = Critic(self.obs_window)

        # self.lr_actor = 0.05  # learning rate for the actor dataframe
        # self.lr_critic = 0.5  # learning rate for the critic dataframe

        # capacity from original FSP implementation from deepmind (as poker hands)
        self.sl_buffer = Reservoir(capacity=40000)
        self.average = AveragePolicy(self.obs_window, self.action_space_list)

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def save_agent(self, name):
        self.actor.model.save_weights(f"./{name}/blue_actor_{name}")
        self.critic.model.save_weights(f"./{name}/blue_critic_{name}")
        self.average.model.save_weights(f"./{name}/blue_average_{name}")

    def load_agent(self, name):
        self.actor.model.load_weights(f"./{name}/blue_actor_{name}")
        self.critic.model.load_weights(f"./{name}/blue_critic_{name}")
        self.average.model.load_weights(f"./{name}/blue_average_{name}")

    def end_episode(self):
        self.last_observation = None
        self.last_policy = None
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass

    def _resolve_action(self, action, session):
        if action[0] == "Analyse":
            return Analyse(hostname=action[1], agent="Blue", session=session)
        elif action[0] == "Remove":
            return Remove(hostname=action[1], agent="Blue", session=session)
        elif action[0] == "Restore":
            return Restore(hostname=action[1], agent="Blue", session=session)
        else:
            return Monitor(agent="Blue", session=session)

    def get_best_action(self, observation, action_space):

        # assume a single session in the action space
        session = list(action_space["session"].keys())[0]

        # get stochastic strategy for this observation
        probs = self.actor.model.predict(
            np.reshape(observation, [1, self.obs_window]), verbose=0
        )
        # select an action from the strategy
        action = np.random.choice(self.action_space_list, p=probs[0])

        self.last_observation = observation
        self.last_policy = probs
        self.last_action = action

        return self._resolve_action(action, session)

    def get_average_action(self, observation, action_space):

        # assume a single session in the action space
        session = list(action_space["session"].keys())[0]

        action = self.average.act(observation)
        return self._resolve_action(action, session)

    def train(self, reward, observation):

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


class RedAC(BaseAgent):
    def __init__(self):

        self.last_observation = (
            None  # remember the observation-action pair used for training
        )
        self.last_action = None

        lone_actions = ["Sleep", "Impact"]  # actions with no parameters
        network_actions = ["DiscoverSystems"]  # actions with a subnet as the parameter
        subnets = "Enterprise", "Op", "User"
        host_actions = (
            "DiscoverServices",
            "ExploitServices",
            "PrivilegeEscalate",
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
        self.action_space_list = (
            lone_actions
            + list(product(network_actions, subnets))
            + list(product(host_actions, hostnames))
        )

        self.actor = pd.DataFrame(columns=self.action_space_list)
        self.critic = pd.DataFrame(columns=self.action_space_list)

        self.subnet_map = {}  # subnets are ordered [Enterprise, Op, User]
        self.ip_map = (
            {}
        )  # ip addresses are ordered [Defender, 'Enterprise0', 'Enterprise1', 'Enterprise2',
        # 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4']

        self.lr_actor = 0.5  # learning rate for the actor dataframe
        self.lr_critic = 0.5  # learning rate for the critic dataframe

    # Red must use the specific IP addresses of targets, instead of hostnames, for actions.
    # At the start of a new episode, these IP addresses must be mapped to hostnames in order
    # to translate the agent's action choices
    def map_network(self, env):
        subnets = "Enterprise", "Op", "User"
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

        i = 0  # count through the networks to assign the correct IP
        for subnet in env.get_action_space(agent="Red")["subnet"]:
            self.subnet_map[subnets[i]] = subnet
            i += 1

        i = 0  # counter through the IP addresses to assign the correct hostname
        for address in env.get_action_space(agent="Red")["ip_address"]:
            if i != 0:
                self.ip_map[hostnames[i - 1]] = address
            i += 1

    def save_agent(self, name):
        self.actor.to_csv(f"red_actor_{name}.csv")
        self.critic.to_csv(f"red_critic_{name}.csv")

    def load_agent(self, name):
        self.actor = pd.read_csv(f"red_actor_{name}.csv", index_col=0)
        self.critic = pd.read_csv(f"red_critic_{name}.csv", index_col=0)

        self.actor.columns = (
            self.action_space_list
        )  # replace loaded strings with tuples
        self.critic.columns = self.action_space_list

    def end_episode(self):
        self.last_observation = None
        self.last_action = None
        self.subnet_map = {}
        self.ip_map = {}

    def set_initial_values(self, action_space, observation):
        pass

    def get_action(self, observation, action_space):

        # Select from all actions, legal or not, rely on environment to fail incorrect actions
        # Train against Blue Sleep agent, then React Remove, then React Restore, and finally against a trained Blue Agent

        # If new observation is not in Actor dataframe, add it
        if str(np.fromstring(observation, dtype=int)) not in self.actor.index:
            # the new policy for this new observation is to select all actions equally
            new_state = pd.DataFrame(
                1,
                columns=self.action_space_list,
                index=[str(np.fromstring(observation, dtype=int))],
            )
            self.actor = self.actor.append(new_state)

        # select an action using the policy for this observation
        select_action = random.uniform(
            0, self.actor.loc[str(np.fromstring(observation, dtype=int))].sum()
        )  # random from 0 to Sum of all values in the row
        find_action = 0
        action = ["Sleep"]
        for column in self.actor:
            find_action += self.actor.loc[str(np.fromstring(observation, dtype=int))][
                column
            ]
            if find_action >= select_action:
                action = column
                break

        # assume a single session in the action space
        session = list(action_space["session"].keys())[0]

        # store this observation-action pair for training
        self.last_observation = observation
        self.last_action = action

        if action[0] == "Impact":
            return Impact(agent="Red", hostname="Op_Server0", session=session)
        elif action[0] == "DiscoverSystems":
            return DiscoverRemoteSystems(
                subnet=self.subnet_map[action[1]], agent="Red", session=session
            )
        elif action[0] == "DiscoverServices":
            return DiscoverNetworkServices(
                ip_address=self.ip_map[action[1]], agent="Red", session=session
            )
        elif action[0] == "ExploitServices":
            return ExploitRemoteService(
                ip_address=self.ip_map[action[1]], agent="Red", session=session
            )
        elif action[0] == "PrivilegeEscalate":
            return PrivilegeEscalate(hostname=action[1], agent="Red", session=session)
        else:
            return Sleep()

    def train(self, reward, observation):

        # if the new state is not in the critic dataframe, add it
        if str(np.fromstring(observation, dtype=int)) not in self.critic.index:
            # initialize the value for every action to the learning rate to encourage exploration but never have 0 or positive value
            new_state = pd.DataFrame(
                self.lr_critic,
                columns=self.action_space_list,
                index=[str(np.fromstring(observation, dtype=int))],
            )
            self.critic = self.critic.append(new_state)

        # update the critic for the previous observation
        q_last = self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][
            self.last_action
        ]
        q_max = max(
            self.critic.loc[str(np.fromstring(observation, dtype=int))]
        )  # value of the best action in the new state

        loss = reward + q_max - q_last

        if loss > 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][
                self.last_action
            ] = (q_last + self.lr_critic)
        elif loss < 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][
                self.last_action
            ] = max(0, (q_last - self.lr_critic))

        # update the actor for the previous observation-action pair
        action_weight = self.actor.loc[
            str(np.fromstring(self.last_observation, dtype=int))
        ][
            self.last_action
        ]  # the current action weight that will be increased
        total_policy_weight = self.actor.loc[
            str(np.fromstring(self.last_observation, dtype=int))
        ].sum()  # the total weights assigned across all actions for the current policy
        adjustment = (
            total_policy_weight - action_weight
        ) / total_policy_weight  # the increase to the weight for this action needs to be adjusted for how likely the action is to
        # be picked. For example, if an action is 90% likely, its increase needs to be adjusted by 1/10
        self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))][
            self.last_action
        ] = action_weight + adjustment * self.lr_actor * (
            reward + q_max
        )  # goes up more for lower regret
