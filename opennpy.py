import numpy as np
import matplotlib.pyplot as plt
import utils as utils

import torch
from gym_torcs import TorcsEnv

rewards = np.load('./buffer_raw/test_buffer_reward.npy')
action = np.load('./buffer_raw/test_buffer_action.npy')
raw_state = np.load('./buffer_raw/test_buffer_raw_state.npy')
raw_next_state = np.load('./buffer_raw/test_buffer_raw_next_state.npy')
#new_rewards = np.load('./buffer_raw/test_buffer_new_reward.npy')
state = np.load('./buffer_raw/test_buffer_state.npy')
next_state = np.load('./buffer_raw/test_buffer_next_state.npy')
not_dones = np.load('./buffers_trained/test_buffer_not_done.npy')
results = np.load('./results/cBCQ_torcs.npy')

print(results)
print(np.count_nonzero(rewards == -1))
print(np.count_nonzero(not_dones == 0))
print(np.where(rewards==-1))

# plt.plot(list(range(len(results))), results, c="b", lw=3, ls="--")
# plt.xlabel("Steps(4000)")
# plt.ylabel("Reward")
# plt.title("Reward Graph")
# fig = plt.gcf()
# plt.show()
# fig.savefig('./results/cBCQ_torcs.png')

#new reward

dones_loc = np.where(not_dones==0)
failed = np.zeros(dones_loc[0].__len__() - 1)
for i in range(0, dones_loc[0].__len__() -1):
    if (dones_loc[0][i+1] - dones_loc[0][i] < 300):
        failed[i] = True
    else:
        failed[i] = False


env = TorcsEnv(vision=False, throttle=True, gear_change=False)

state_dim = 29  # state의 생김새
action_dim = 3  # action의 갯수
max_action = env.action_space.high[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

success_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
failed_buffer = utils.ReplayBuffer(state_dim, action_dim, device)


counter = 0
for i in range(0, 96882):
    if(not_dones[i]==0):
        counter+=1

    if(counter in np.where(failed==1)[0]):
        failed_buffer.add(state[i], action[i], next_state[i], rewards[i], 1.- not_dones[i] , raw_state[i], raw_next_state[i])
    elif(counter in np.where(failed==0)[0]):
        success_buffer.add(state[i], action[i], next_state[i], rewards[i], 1. - not_dones[i], raw_state[i], raw_next_state[i])

failed_buffer.save(f"./failed_buffer/failed_buffer")
success_buffer.save(f"./success_buffer/success_buffer")


#state : ob.angle (1), ob.track(19), ob.trackPos(1), ob.speedY(1), ob.speedX(1), ob.speedZ(1), ob.wheelSpinVel(4), ob.rpm(1)
def new_reward_func(save_folder):
    max_size = int(1e5)
    obs_array = np.load(f"{save_folder}_raw_state.npy")
    not_dones =np.load(f"{save_folder}_not_done.npy")
    new_reward_buffer = np.zeros((max_size, 1))

    count = 0
    for i in range (obs_array.shape[0]):
        angle = obs_array[i][0]
        track = obs_array[i][1:20]
        trackPos = np.abs(obs_array[i][20])
        speedY = obs_array[i][21]
        speedX = obs_array[i][22]
        speedZ = obs_array[i][23]
        wheelSpinVel = obs_array[i][24:28]
        rpm = obs_array[28]

        #progress = speedX * np.cos(angle)

        #sp =speedX/8.0
        #progress = sp * np.cos(angle) - trackPos * 30

        # sp = speedX
        # progress = sp *np.cos(angle) + sp * 2.5

        sp = speedX
        progress = sp *np.cos(angle)

        new_reward = progress

        if not_dones[i] == 0:
            new_reward = -1
            count +=1

        new_reward_buffer[i] = new_reward

        if i % 10000 == 0:
            print(i)

    np.save(f"{save_folder}_reward.npy", new_reward_buffer[:new_reward_buffer.size])

buffer_name = 'test_buffer'
new_reward_func(f"./buffer_original_reward2/{buffer_name}")
# Termination judgement #########################
# episode_terminate = False

names = ['focus'#0~4
         'speedX'#5
         'speedY',#6
         'speedZ',#7
         'angle',#8
         'opponents',#
         'rpm',
         'track',
         'trackPos',
         'wheelSpinVel',
         'img']