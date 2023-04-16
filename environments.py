from CybORG import CybORG
from wrapper import CompetitiveWrapper

import gym
from gym.spaces import Discrete, MultiBinary

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env

import numpy as np
from itertools import product
from random import randint, random
from scipy.special import softmax
from statistics import mean

timesteps = 12 # per game

gae = 1
gamma = 0.99
epochs = 30
mixer = 0.9 # for training opponent best-response, how many games with current agent policy instead of agent pool

red_batch_size = 61440
red_minibatch_size = 3840
red_lr = 1e-3
red_entropy = 1e-3
red_kl = 1
red_clip_param = 0.5

blue_batch_size = 61440
blue_minibatch_size = 3840
blue_lr = 1e-3
blue_entropy = 1e-3
blue_kl = 1
blue_clip_param = 0.5

layer_units = 256
model_arch = [layer_units, layer_units]
act_func = "tanh"

experiment_name = "phase1"
# construct the blue and red action spaces
subnets = "Op", "User"
hostnames = (
    "Op_Host0",
    "Op_Server0",
    "User1",
    "User2",
    "User3",
)
blue_lone_actions = [["Monitor"]]  # actions with no parameters
blue_host_actions = (
    "Analyse",
    "Remove",
    "Restore",
)  # actions with a hostname parameter
red_lone_actions = [["Sleep"], ["Impact"]]  # actions with no parameters
red_network_actions = [
    "DiscoverSystems"
]  # actions with a subnet as the parameter
red_host_actions = (
    "DiscoverServices",
    "ExploitServices",
    "PrivilegeEscalate",
)
blue_action_list = blue_lone_actions + list(
    product(blue_host_actions, hostnames)
)
red_action_list = (
    red_lone_actions
    + list(product(red_network_actions, subnets))
    + list(product(red_host_actions, hostnames))
)
blue_obs_space = 5*len(hostnames) + timesteps + 1
red_obs_space = 29 + timesteps + 1

class BlueTrainer(gym.Env):
    def __init__(self, env_config):

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(red_obs_space)
        config["action_space"] = Discrete(len(red_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.red_opponent = PPO(config=config)
        self.opponent_id = 0
        
    def reset(self):

        pool_file = open("./policies/red_opponent_pool/pool_size", "r")
        red_pool_size = int(pool_file.read())
        pool_file.close()

        if red_pool_size > 0:
            self.opponent_id = randint(1, red_pool_size)
        else:
            self.opponent_id = 0
        
        path_file = open(f"./policies/red_opponent_pool/opponent_red_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        info = {}
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True  

        return (obs, reward, done, info)
    

class BlueOpponent(gym.Env):
    def __init__(self, env_config):

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(red_obs_space)
        config["action_space"] = Discrete(len(red_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.red_opponent = PPO(config=config)
        self.opponent_id = 0
        
    def reset(self):

        pool_file = open("./policies/red_competitive_pool/pool_size", "r")
        red_pool_size = int(pool_file.read())
        pool_file.close()

        if ((red_pool_size > 1) and (random() > mixer)):
            self.opponent_id = randint(1, red_pool_size-1)
        else:
            self.opponent_id = red_pool_size
        
        path_file = open(f"./policies/red_competitive_pool/competitive_red_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        info = {}
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True  

        return (obs, reward, done, info)


class DedicatedBlueEnv(gym.Env):
    def __init__(self, env_config):

        # agent name, for saving and loading
        self.name = "blue_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the blue action and observation spaces as gym objects
        self.action_space = Discrete(len(blue_action_list))
        self.observation_space = MultiBinary(blue_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return  observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # Red config for opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(red_obs_space)
        config["action_space"] = Discrete(len(red_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.red_opponent = PPO(config=config)
        
    def reset(self):

        path_file = open(f"./policies/competitive_red_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.red_opponent.restore(checkpoint_path)
    
        obs, self.red_obs = self.cyborg.reset()    
        return obs
    
    # the step function should receive a blue action
    # the red action will be chosen within the step function
    def step(self, action, verbose=False):

        red_action = self.red_opponent.compute_single_action(self.red_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=red_action,
            blue_action=action
        )

        # blue reward and new observation
        obs = state.blue_observation
        reward = state.reward
        self.red_obs = state.red_observation

        # show actions and new observation if examining a trained policy
        if verbose == True:
            print(f"Blue Action: {blue_action_list[action]}")
            print(f"Red Action: {red_action_list[red_action]}")
            print(self.cyborg.get_blue_table())
            print(obs)
            print(self.cyborg._create_red_table())
            print(self.red_obs)
            print(f'Known Subnets: {self.cyborg.known_subnets}')
                    
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)
    
class RedTrainer(gym.Env):
    def __init__(self, env_config):
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(blue_obs_space)
        config["action_space"] = Discrete(len(blue_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.blue_opponent = PPO(config=config)
        self.opponent_id = 0

    def reset(self):

        pool_file = open("./policies/blue_opponent_pool/pool_size", "r")
        blue_pool_size = int(pool_file.read())
        pool_file.close()

        if blue_pool_size > 0:
            self.opponent_id = randint(1, blue_pool_size)
        else:
            self.opponent_id = blue_pool_size
        
        path_file = open(f"./policies/blue_opponent_pool/opponent_blue_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)
    
        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)


class RedOpponent(gym.Env):
    def __init__(self, env_config):
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(blue_obs_space)
        config["action_space"] = Discrete(len(blue_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.blue_opponent = PPO(config=config)
        self.opponent_id = 0

    def reset(self):

        pool_file = open("./policies/blue_competitive_pool/pool_size", "r")
        blue_pool_size = int(pool_file.read())
        pool_file.close()

        if ((blue_pool_size > 1) and (random() > mixer)):
            self.opponent_id = randint(1, blue_pool_size-1)
        else:
            self.opponent_id = blue_pool_size
        
        path_file = open(f"./policies/blue_competitive_pool/competitive_blue_{self.opponent_id}/checkpoint_path", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)
    
        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)


class DedicatedRedEnv(gym.Env):
    def __init__(self, env_config):
        self.name = "red_env"

        # max timesteps per episode
        self.max_t = timesteps

        # define the red action and observation spaces as gym objects
        self.action_space = Discrete(len(red_action_list))
        self.observation_space = MultiBinary(red_obs_space)

        # create a CybORG environment with no agents (neither agent will be controlled by the environment)
        cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
        # wrapper to accept red and blue actions, and return red observations
        self.cyborg = CompetitiveWrapper(turns=timesteps, env=cyborg, output_mode="vector")

        # copy of a config for the Blue opponent
        config = {}
        config["num_workers"] = 0
        config["model"] = {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers": False}
        config["observation_space"] = MultiBinary(blue_obs_space)
        config["action_space"] = Discrete(len(blue_action_list))
        config['vf_share_layers'] = False
        config['log_sys_usage'] = False
        self.blue_opponent = PPO(config=config)

    def reset(self):

        path_file = open(f"./policies/competitive_blue_policy", "r")
        checkpoint_path = path_file.read()
        path_file.close()
        self.blue_opponent.restore(checkpoint_path)

        self.blue_obs, obs = self.cyborg.reset()   
        return obs

    # the step function should receive a red action
    # the blue action will be chosen within the step function
    def step(self, action, verbose=False):

        blue_action = self.blue_opponent.compute_single_action(self.blue_obs)

        # advance to the new state
        state = self.cyborg.step(
            red_action=action,
            blue_action=blue_action
        )

        # red reward and new observation
        obs = state.red_observation
        reward = -state.reward # reward signal is flipped here for the red agent
        self.blue_obs = state.blue_observation
  
        # episode is done if last timestep has been reached
        done = False
        if self.cyborg.turn == self.cyborg.turns_per_game:
            done = True
        
        info = {}

        return (obs, reward, done, info)


def build_blue_agent(opponent=False, dedicated=False, workers=40, fresh=True):
    # register the custom environment
    if dedicated:
        select_env = "DedicatedBlueEnv"
        register_env(
            select_env,
            lambda config: DedicatedBlueEnv(
                env_config={"name": f"{experiment_name}_dedicated_blue"}
            )
        )
    elif opponent:
        select_env = "BlueOpponent"
        register_env(
            select_env,
            lambda config: BlueOpponent(
                env_config={"name": f"{experiment_name}_opponent_blue"}
            )
        )
    else:
        select_env = "BlueTrainer"
        register_env(
            select_env,
            lambda config: BlueTrainer(
                env_config={"name": f"{experiment_name}_competitive_blue"}
            )
        )

    # set the RLLib configuration
    blue_config = {
        "env": "blue_trainer",
        "num_gpus": 1,
        "num_workers": workers,
        "train_batch_size": blue_batch_size,
        "sgd_minibatch_size": blue_minibatch_size,
        'rollout_fragment_length': int(blue_batch_size/workers),
        'num_sgd_iter': epochs,
        'batch_mode': "truncate_episodes",
        "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
        "lr": blue_lr,
        "entropy_coeff": blue_entropy,
        "observation_space": MultiBinary(blue_obs_space),
        "action_space": Discrete(len(blue_action_list)),
        "recreate_failed_workers": True,
        'vf_share_layers': False,
        'lambda': gae,
        'gamma': gamma,
        'kl_coeff': blue_kl,
        'kl_target': 0.01,
        'clip_rewards': False,
        'clip_param': blue_clip_param,
        'vf_clip_param': 50.0,
        'vf_loss_coeff': 0.01,
        'log_sys_usage': False,
        'disable_env_checking': True,
    }

    if dedicated:
        blue_agent = PPO(config=blue_config, env=DedicatedBlueEnv)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/blue_dedicated_pool/dedicated_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/blue_dedicated_pool/dedicated_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open("./policies/blue_dedicated_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    elif opponent:
        blue_agent = PPO(config=blue_config, env=BlueOpponent)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/blue_opponent_pool/opponent_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/blue_opponent_pool/opponent_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open("./policies/blue_opponent_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    else:
        blue_agent = PPO(config=blue_config, env=BlueTrainer)
        if fresh:
            checkpoint_path = blue_agent.save(checkpoint_dir=f"./policies/blue_competitive_pool/competitive_blue_0")
            print(checkpoint_path)
            path_file = open(f"./policies/blue_competitive_pool/competitive_blue_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            path_file = open("./policies/blue_competitive_pool/pool_size", "w")
            path_file.write("0")
            path_file.close() 
    return blue_agent

def build_red_agent(opponent=False, dedicated=False, workers=40, fresh=True):
    # register the custom environment
    if dedicated:
        select_env = "DedicatedRedEnv"
        register_env(
            select_env,
            lambda config: DedicatedRedEnv(
                env_config={"name": f"{experiment_name}_dedicated_red"}
            )
        )
    elif opponent:
        select_env = "RedOpponent"
        register_env(
            select_env,
            lambda config: RedOpponent(
                env_config={"name": f"{experiment_name}_opponent_red"}
            )
        )
    else:
        select_env = "RedTrainer"
        register_env(
            select_env,
            lambda config: RedTrainer(
                env_config={"name": f"{experiment_name}_competitive_red"}
            )
        )

    # set the RLLib configuration
    red_config = {
        "env": "RedTrainer",
        "num_gpus":  1,
        "num_workers": workers,
        "train_batch_size": red_batch_size,
        "sgd_minibatch_size": red_minibatch_size,
        'rollout_fragment_length': int(red_batch_size/workers),
        'num_sgd_iter': epochs,
        'batch_mode': "truncate_episodes",
        "model": {"fcnet_hiddens": model_arch, "fcnet_activation": act_func, "vf_share_layers":False},
        "lr": red_lr,
        "entropy_coeff": red_entropy,
        "observation_space": MultiBinary(red_obs_space),
        "action_space": Discrete(len(red_action_list)),
        "recreate_failed_workers": True,
        'vf_share_layers': False,
        'lambda': gae,
        'gamma': gamma,
        'kl_coeff': red_kl,
        'kl_target': 0.01,
        'clip_rewards': False,
        'clip_param': red_clip_param,
        'vf_clip_param': 50.0,
        'vf_loss_coeff': 0.01,
        'log_sys_usage': False,
        'disable_env_checking': True,
    }

    if dedicated:
        red_agent = PPO(config=red_config, env=DedicatedRedEnv)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/red_dedicated_pool/dedicated_red_0")
            path_file = open(f"./policies/red_dedicated_pool/dedicated_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open("./policies/red_dedicated_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    elif opponent:
        red_agent = PPO(config=red_config, env=RedOpponent)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/red_opponent_pool/opponent_red_0")
            path_file = open(f"./policies/red_opponent_pool/opponent_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open("./policies/red_opponent_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()
    else:
        red_agent = PPO(config=red_config, env=RedTrainer)
        if fresh:
            checkpoint_path = red_agent.save(checkpoint_dir=f"./policies/red_competitive_pool/competitive_red_0")
            path_file = open(f"./policies/red_competitive_pool/competitive_red_0/checkpoint_path", "w")
            path_file.write(checkpoint_path)
            path_file.close()
            print(checkpoint_path)
            path_file = open("./policies/red_competitive_pool/pool_size", "w")
            path_file.write("0")
            path_file.close()     
    return red_agent

def sample(test_red, test_blue, games=1, verbose=False, show_policy=False, blue_moves=None, red_moves=None, random_blue=False, random_red=False):
    base_cyborg = CybORG(scenario_file="./scenario.yaml", environment="sim", agents=None)
    # wrapper to accept red and blue actions, and return  observations
    cyborg = CompetitiveWrapper(env=base_cyborg, turns=timesteps, output_mode="vector")

    scores = []
    max_score = 0
    min_score = float('inf')

    for g in range(games):

        blue_obs, red_obs = cyborg.reset()
        score = 0

        if verbose and (games>1):
            print(f"-------- Game {g+1} --------")
        
        if random_blue:
            blue_moves = []
            for t in range(timesteps):
                blue_moves.append(randint(1, len(blue_action_list)-1))
        
        if random_red:
            red_moves = []
            for t in range(timesteps):
                red_moves.append(randint(1, len(red_action_list)-1))

        for t in range(timesteps):

            if blue_moves is None:
                blue_action, _, blue_extras = test_blue.compute_single_action(blue_obs, full_fetch=True)
            else:
                blue_action = blue_moves[t]
                blue_extras = {'action_dist_inputs':np.zeros(len(blue_action_list)), 'action_prob':1}

            if red_moves is None:
                red_action, _, red_extras = test_red.compute_single_action(red_obs, full_fetch=True)
            else:
                red_action = red_moves[t]
                red_extras = {'action_dist_inputs':np.zeros(len(red_action_list)), 'action_prob':1}

            state = cyborg.step(red_action, blue_action)

            red_reward = -state.reward

            blue_obs = state.blue_observation
            red_obs = state.red_observation

            score += red_reward

            if verbose:
                if 'policy' in blue_extras:
                    blue_policy = blue_extras['policy']
                else:
                    blue_policy = softmax(blue_extras['action_dist_inputs'])
                if 'policy' in red_extras:
                    red_policy = red_extras['policy']
                else:
                    red_policy = softmax(red_extras['action_dist_inputs'])

                print(f'---- Turn {t+1} ----')
                if show_policy:
                    print("Blue policy: ")
                    for a in range(len(blue_action_list)):
                        print(f"{blue_action_list[a]}: {blue_policy[a]*100:0.2f}%")
                print(f"Blue selects {blue_action_list[blue_action]} with probability {blue_extras['action_prob']*100:0.2f}%")
                print()
                if show_policy:
                    print(f"Red Policy: ")
                    for a in range(len(red_action_list)):
                        print(f"{red_action_list[a]}: {red_policy[a]*100:0.2f}%")
                print(f"Red selects {red_action_list[red_action]} with probability {red_extras['action_prob']*100:0.2f}%")
                print()
                print(f'New Red observation: {red_obs}')
                print(cyborg._create_red_table())
                print()
                print(f"Reward: +{red_reward:0.1f}")
                print(f"Score: {score:0.1f}")
                print()
        
        scores.append(score)
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
    
    avg_score = mean(scores)
    if verbose and (games>1):
        print(f'Average Score for {games} Games is {avg_score}')
        print(f'High Score is {max_score}')
        print(f'Low Score is {min_score}')
    
    return(avg_score)


