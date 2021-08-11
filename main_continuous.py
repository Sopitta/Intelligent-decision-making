import glob
import os
import sys
import random
import time
import numpy as np
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import matplotlib.pyplot as plt

try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('Z:/Documents/Carla/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('C:/School/Carla sim/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
#from env import CarEnv, World, HUD
from Env.env_continuous import World, HUD
#from agent.myagent import Agent
from high_level_sc import HighLevelSC
from stable_baselines import DQN #get action from DQN and evn.step(action)
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.schedules import LinearSchedule


#train = True
#training_number = 4
#train_time_step = 12000000
#method = 'dqn'
#log_fold = "./sopitta_logs_{}/".format(method)
#model_name = "./sopitta_logs_{}/{}_{}".format(method, method, training_number)
#log_name = "log_{}_{}".format(method, training_number)
#env = World()

def train_model(env_0, log_dir, log_name, train_num, model_name, train_time, load = False):
    #callback = TensorboardCallback(env=env_0)
    # Input features and reward normalization
    #print(load)
    env = DummyVecEnv([lambda: env_0])
    if load:
        stats_path_prev = os.path.join(log_dir, "vec_normalize_{}.pkl".format(train_num-1))
        env = VecNormalize.load(stats_path_prev, env)
    else: #train from scratch
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0,
                           clip_reward=10.0,
                           gamma=0.99,
                           epsilon=1e-08)
    # Custom MLP policy of three layers of size 128 each with tanh activation function
    # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128, 128])
    lr_schedule = LinearSchedule(train_time, final_p=1e-3, initial_p=1e-5)
    policy_kwargs = dict(net_arch=[128, 128, 128])
    model = PPO2(policy=MlpPolicy, env=env, verbose=1, tensorboard_log=log_dir,
                 policy_kwargs=policy_kwargs,
                 gamma=0.99,            # discount factor [0.8 0.99] 0.99
                 n_steps=5000,           #!! horizon [32 5000] [64 2048] 128
                 ent_coef=0.01,          # entropy coefficient [0 0.001] 0.01
                 learning_rate=lr_schedule.value,    #!! learning rate [1e-3 1e-6] 2.5e-4
                 vf_coef=0.5,           # value function coefficient [0.5 1] 0.5
                 max_grad_norm=0.5,     # [] 0.5
                 lam=0.9,               # [0.9 1] 0.9
                 nminibatches=4,        #!! minibatch [4 4096] con [512 5120], des [32 512] 4
                 noptepochs=4,          #!! epoch [3 30] 4
                 cliprange=0.2)         #!! clipping [0.1 0.3] 0.2
    #model.learn(total_timesteps=train_time, tb_log_name=log_name, callback=callback)
    model.learn(total_timesteps=train_time, tb_log_name=log_name)
    model.save(model_name)
    stats_path = os.path.join(log_dir, "vec_normalize_{}.pkl".format(train_num))
    env.save(stats_path)
    #print("Done training, total episodes executed = {}".format(env_0.total_epis))
    print("Done training, total steps executed = {}".format(train_time))
    plt.plot(env_0.cum_r)
    plt.savefig('average_reward_'+str(train_num)+'.png')
    plt.close()
    r_arr = np.array(env_0.cum_r)
    np.save('reward_per_ep_'+str(train_num)+'.npy', r_arr)
    col_arr = np.array(env_0.collision_num)
    np.save('col_num_per_ep_'+str(train_num)+'.npy', col_arr)
    em_arr = env_0.em_num_list
    np.save('em_per_ep_'+str(train_num)+'.npy', em_arr)
    plt.plot(env_0.em_num_list)
    plt.savefig('emergency_break_'+str(train_num)+'.png')
    plt.close()

def evaluate_model(env, model_name, eval_step, log_dir, train_num):
    # EVALUATE
    model = PPO2.load(model_name)
    # Load the saved statistics
    env = DummyVecEnv([lambda: env])
    stats_path = os.path.join(log_dir, "vec_normalize_{}.pkl".format(train_num))
    env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False
    obs = env.reset()
    for i in range(eval_step):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    print('Done evaluation')

# os.environ["SDL_VIDEODRIVER"] = "dummy"

def main():
    train = True
    load = False
    train_num = 29
    method = 'ppo'
    continuous = True
    log_dir = "./{}/".format(method)
    os.makedirs(log_dir, exist_ok=True)
    model_name = "./{}/{}_WalkerCross_{}".format(method, method, train_num)
    #env = World(continuous)
    env = World()
    if train:
        log_name = "log_{}_WalkerCross_{}".format(method, train_num)
        steps = 15000
        train_model(env, log_dir, log_name, train_num, model_name, steps, load = load)
    else:
        steps = 4500
        evaluate_model(env, model_name, steps, log_dir, train_num)
    #env.destroy_all()
    #env.quit_pygame()

if __name__ == '__main__':
    main()
        
    





