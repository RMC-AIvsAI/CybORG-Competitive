from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import Monitor, Analyse, Remove, Restore, \
    PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep
import pandas as pd
import numpy as np
from itertools import product
import random
from ast import literal_eval as make_tuple

class RedAC(BaseAgent):
    def __init__(self):

        self.last_observation = None    # remember the observation-action pair used for training
        self.last_action = None

        lone_actions = ['Sleep','Impact']  # actions with no parameters
        network_actions = ['DiscoverSystems']    # actions with a subnet as the parameter
        subnets = 'Enterprise','Op','User'
        host_actions = 'DiscoverServices', 'ExploitServices', 'PrivilegeEscalate' # actions with a hostname parameter
        hostnames = 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4'
        self.action_space_list = lone_actions + list(product(network_actions,subnets)) + list(product(host_actions,hostnames))

        self.actor = pd.DataFrame(columns=self.action_space_list)
        self.critic = pd.DataFrame(columns=self.action_space_list)

        self.subnet_map = {} # subnets are ordered [Enterprise, Op, User]
        self.ip_map = {} # ip addresses are ordered [Defender, 'Enterprise0', 'Enterprise1', 'Enterprise2',
                            # 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4']
        
        self.lr_actor = 0.5     # learning rate for the actor dataframe
        self.lr_critic = 0.5   # learning rate for the critic dataframe
    
    # Red must use the specific IP addresses of targets, instead of hostnames, for actions.
    # At the start of a new episode, these IP addresses must be mapped to hostnames in order
    # to translate the agent's action choices
    def map_network(self, env):
        subnets = 'Enterprise','Op','User'
        hostnames = 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4'

        i=0 # count through the networks to assign the correct IP
        for subnet in env.get_action_space(agent='Red')["subnet"]:
            self.subnet_map[subnets[i]] = subnet
            i+=1

        i=0 # counter through the IP addresses to assign the correct hostname
        for address in env.get_action_space(agent='Red')["ip_address"]:
            if i != 0:
                self.ip_map[hostnames[i-1]] = address
            i+=1
    
    def save_agent(self, name):
        self.actor.to_csv(f'red_actor_{name}.csv')
        self.critic.to_csv(f'red_critic_{name}.csv')

    def load_agent(self, name):
        self.actor = pd.read_csv(f'red_actor_{name}.csv', index_col=0)
        self.critic = pd.read_csv(f'red_critic_{name}.csv', index_col=0)
        
        self.actor.columns = self.action_space_list # replace loaded strings with tuples
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
            new_state = pd.DataFrame(1, columns=self.action_space_list, index=[str(np.fromstring(observation, dtype=int))])
            self.actor = self.actor.append(new_state)

        # select an action using the policy for this observation
        select_action = random.uniform(0,self.actor.loc[str(np.fromstring(observation, dtype=int))].sum()) # random from 0 to Sum of all values in the row
        find_action = 0
        action = ['Sleep']
        for column in self.actor:
            find_action += self.actor.loc[str(np.fromstring(observation, dtype=int))][column]
            if find_action >= select_action:
                action = column
                break

        # assume a single session in the action space
        session = list(action_space['session'].keys())[0]

        # store this observation-action pair for training
        self.last_observation = observation
        self.last_action = action

        if action[0] == 'Impact':
            return Impact(agent='Red', hostname='Op_Server0', session=session)
        elif action[0] == 'DiscoverSystems':
            return DiscoverRemoteSystems(subnet=self.subnet_map[action[1]], agent='Red', session=session)
        elif action[0] == 'DiscoverServices':
            return DiscoverNetworkServices(ip_address=self.ip_map[action[1]], agent='Red', session=session)
        elif action[0] == 'ExploitServices':
            return ExploitRemoteService(ip_address=self.ip_map[action[1]], agent='Red', session=session)
        elif action[0] == 'PrivilegeEscalate':
            return PrivilegeEscalate(hostname=action[1], agent='Red', session=session)
        else:
            return Sleep()

    def train(self, reward, observation):
        
        # if the new state is not in the critic dataframe, add it
        if str(np.fromstring(observation, dtype=int)) not in self.critic.index:
            # initialize the value for every action to the learning rate to encourage exploration but never have 0 or positive value
            new_state = pd.DataFrame(self.lr_critic, columns=self.action_space_list, index=[str(np.fromstring(observation, dtype=int))])
            self.critic = self.critic.append(new_state)

        # update the critic for the previous observation
        q_last = self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action]
        q_max = max(self.critic.loc[str(np.fromstring(observation, dtype=int))])   # value of the best action in the new state
        
        loss = reward + q_max - q_last

        if loss > 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = q_last + self.lr_critic
        elif loss < 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = max(0,(q_last-self.lr_critic))

        # update the actor for the previous observation-action pair
        action_weight = self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] # the current action weight that will be increased
        total_policy_weight = self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))].sum() # the total weights assigned across all actions for the current policy
        adjustment = (total_policy_weight - action_weight)/total_policy_weight # the increase to the weight for this action needs to be adjusted for how likely the action is to 
                        # be picked. For example, if an action is 90% likely, its increase needs to be adjusted by 1/10
        self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = action_weight + adjustment*self.lr_actor*(reward + q_max) # goes up more for lower regret


class BlueAC(BaseAgent):
    def __init__(self):
        self.host_list = []
        self.last_observation = None    # remember the observation-action pair used for training
        self.last_action = None
        self.timesteps = 15         # this agent is being trained for a game of length 10

        lone_actions = ['Monitor']  # actions with no parameters
        host_actions = 'Analyse', 'Remove', 'Restore' # actions with a hostname parameter
        hostnames = 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4'
        self.action_space_list = lone_actions + list(product(host_actions,hostnames))
        
        self.actor = pd.DataFrame(columns=self.action_space_list) # rows are states, columns are actions, cells hold probabilities for each state-action
        self.critic = pd.DataFrame(columns=self.action_space_list) # rows are states, columns are remaining timesteps, cells hold state value for each timestep

        self.lr_actor = 0.05    # learning rate for the actor dataframe
        self.lr_critic = 0.5   # learning rate for the critic dataframe

    def save_agent(self, name):
        self.actor.to_csv(f'blue_actor_{name}.csv')
        self.critic.to_csv(f'blue_critic_{name}.csv')

    def load_agent(self, name):
        self.actor = pd.read_csv(f'blue_actor_{name}.csv', index_col=0)
        self.critic = pd.read_csv(f'blue_critic_{name}.csv', index_col=0)
        
        self.actor.columns = self.action_space_list # replace loaded strings with tuples
        self.critic.columns = self.action_space_list
    
    def end_episode(self):
        self.host_list = []
        self.last_observation = None
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass

    def get_action(self, observation, action_space):

        # add suspicious hosts to the hostlist if monitor found something
        # added line to allow for automatic monitoring.
        # if self.last_action is not None and self.last_action == 'Monitor':
        #     for host_name, host_info in [(value['System info']['Hostname'], value) for key, value in observation.items() if key != 'success']:
        #         if host_name not in self.host_list and host_name != 'User0' and 'Processes' in host_info and len([i for i in host_info['Processes'] if 'PID' in i]) > 0:
        #             self.host_list.append(host_name)

        # if the new state is not in the actor dataframe, add it
        if str(np.fromstring(observation, dtype=int)) not in self.actor.index:
            # the new policy for this new observation is to select all actions equally
            new_state = pd.DataFrame(1, columns=self.action_space_list, index=[str(np.fromstring(observation, dtype=int))])
            self.actor = self.actor.append(new_state)

        # select an action using the policy for this observation
        select_action = random.uniform(0,self.actor.loc[str(np.fromstring(observation, dtype=int))].sum()) # random from 0 to Sum of all values in the row
        find_action = 0
        action = ['Monitor']
        for column in self.actor:
            find_action += self.actor.loc[str(np.fromstring(observation, dtype=int))][column]
            if find_action >= select_action:
                action = column
                break

        # assume a single session in the action space
        session = list(action_space['session'].keys())[0]
        
        # store this observation-action pair for training
        self.last_observation = observation
        self.last_action = action

        if action[0] == 'Analyse':
            return Analyse(hostname=action[1], agent='Blue', session=session)
        elif action[0] == 'Remove':
            return Remove(hostname=action[1], agent='Blue', session=session)
        elif action[0] == 'Restore':
            return Restore(hostname=action[1], agent='Blue', session=session)
        else:
            return Monitor(agent='Blue', session=session)
    
    def train(self, reward, observation):
        
        # if the new state is not in the critic dataframe, add it
        if str(np.fromstring(observation, dtype=int)) not in self.critic.index:
            # initialize the value for every action to the learning rate to encourage exploration but never have 0 or positive value
            new_state = pd.DataFrame(-(self.lr_critic), columns=self.action_space_list, index=[str(np.fromstring(observation, dtype=int))])
            self.critic = self.critic.append(new_state)

        # update the critic for the previous observation
        q_last = self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action]
        q_max = max(self.critic.loc[str(np.fromstring(observation, dtype=int))])   # value of the best action in the new state
        
        loss = reward + q_max - q_last

        if loss > 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = min(0, (q_last+self.lr_critic))
        elif loss < 0:
            self.critic.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = q_last - self.lr_critic

        # update the actor for the previous observation-action pair
        action_weight = self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] # the current action weight that will be increased
        total_policy_weight = self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))].sum() # the total weights assigned across all actions for the current policy
        adjustment = total_policy_weight/action_weight
            # TRYING THIS INSTEAD, to counter the fact that each action is samples action_weight/total_policy_weight 
                        # the increase to the weight for this action needs to be adjusted for how likely the action is to 
                        # be picked. For example, if an action is 90% likely, its increase needs to be adjusted by 1/10
        self.actor.loc[str(np.fromstring(self.last_observation, dtype=int))][self.last_action] = \
            action_weight + adjustment*self.lr_actor/-(min(-0.1,(reward+q_max))) # goes up more for lower regret
                                                         # min here to stop division by 0, makes the fraction a max of 1