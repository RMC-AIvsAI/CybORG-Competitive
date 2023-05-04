This repository contains the source code used for the thesis: Competitive Reinforcement Learning for Autonomous Cyber Operations.  
  
The CybORG directory is taken from the CAGE Challenge 2 CybORG repository, which can be found at: https://github.com/cage-challenge. The EnvionmentController.py and Results.py files have been modified to accept both Red and Blue agent actions simultaneously, and return observations for both agents. The installation instructions included in the CAGE Challenge release are still required.  
  
The environments.py file includes new environments that are used to train Blue and Red agent policies using RLLib's PPO algorithm: https://docs.ray.io/en/latest/rllib/index.html. It also includes the functions used to create new learning agents, and produce sample games between trained agents.  
  
wrapper.py includes a new competitive wrapper, which is based on the Blue Table Wrapper that was included in the CAGE Challenge release. This competitive wrapper incorporates new observation vectors for both agents that are unique for the thesis scenario, and includes new methods to resolve each agents discrete action selection. This thesis scenario, which is discussed in the main document, is defined in the scenario.yaml file.  
  
The fictitious play loop that is used to train the competitive policies takes place in the experiment.ipynb notebook. This is also where the exploitability of each agent is measured, relative to the other agents in their policy pool. All policies produced during the experiment are saved in the policies directory. The different pools in this directory each correspond to a training environment from environments.py, and are all used for the experiment as described in the thesis document.  
  
The validation.ipynb notebook is used to train the dedicated opponents that confirm the minmax scores of the trained competitive agents. Finally, the score_martix.ipynb notebook is used to examine the average scores for games played by the various combinations of trained agents.  
