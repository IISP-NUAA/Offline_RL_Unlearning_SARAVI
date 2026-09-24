import gymnasium as gym
import minari
import subprocess
import concurrent.futures
import os
required_dataset_name = 'antmaze'
settings = []
if 'antmaze' in required_dataset_name:
    settings = ['medium-play-v1','umaze-diverse-v1','large-diverse-v1','large-play-v1 ','medium-diverse-v1','umaze-v1']
    sub_dir_name = 'D4RL'
#with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

for setting in settings:
    command = 'minari download '+ sub_dir_name+'/'+required_dataset_name+'/'+setting
    os.system(command=command)