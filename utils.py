import numpy as np
import torch



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e5)): #1e6
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.raw_state = np.zeros((max_size, state_dim))
        self.raw_next_state = np.zeros((max_size, state_dim))

        self.next_state1 = np.zeros((max_size, state_dim))
        self.next_state2 = np.zeros((max_size, state_dim))
        self.next_state3 = np.zeros((max_size, state_dim))
        self.next_state4 = np.zeros((max_size, state_dim))
        self.next_state5 = np.zeros((max_size, state_dim))
        self.next_state6 = np.zeros((max_size, state_dim))
        self.next_state7 = np.zeros((max_size, state_dim))
        self.next_state8 = np.zeros((max_size, state_dim))
        self.next_state9 = np.zeros((max_size, state_dim))
        self.next_state10 = np.zeros((max_size, state_dim))
        self.next_state_total = [self.next_state1, self.next_state2, self.next_state3, self.next_state4, self.next_state5,
                            self.next_state6, self.next_state7, self.next_state8, self.next_state9, self.next_state10]

        self.device = device

    def add(self, state, action, next_state, reward, done , raw_state, raw_next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.raw_state[self.ptr] = raw_state
        self.raw_next_state[self.ptr] = raw_next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_aug(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),

            (torch.FloatTensor(self.next_state1[ind]).to(self.device),
            torch.FloatTensor(self.next_state2[ind]).to(self.device),
            torch.FloatTensor(self.next_state3[ind]).to(self.device),
            torch.FloatTensor(self.next_state4[ind]).to(self.device),
            torch.FloatTensor(self.next_state5[ind]).to(self.device),
            torch.FloatTensor(self.next_state6[ind]).to(self.device),
            torch.FloatTensor(self.next_state7[ind]).to(self.device),
            torch.FloatTensor(self.next_state8[ind]).to(self.device),
            torch.FloatTensor(self.next_state9[ind]).to(self.device),
            torch.FloatTensor(self.next_state10[ind]).to(self.device))
        )

    def sample_episode(self, n_th):

        done = 1 - self.not_done
        done_idx = np.where(done)[0]

        start_point = done_idx[n_th]
        end_point = done_idx[n_th + 1] - 1
        num_sample = end_point - start_point + 1
        ind = np.array(list(range(start_point, end_point + 1)))

        state = np.zeros(((num_sample, self.state_history) + self.state.shape[1:]), dtype=np.uint8)
        next_state = np.array(state)

        state_not_done = 1.
        next_not_done = 1.
        for i in range(self.state_history):

            # Wrap around if the buffer is filled
            if self.crt_size == self.max_size:
                j = (ind - i) % self.max_size
                k = (ind - i + 1) % self.max_size
            else:
                j = ind - i
                k = (ind - i + 1).clip(min=0)
                # If j == -1, then we set state_not_done to 0.
                state_not_done *= (j + 1).clip(min=0, max=1).reshape(-1, 1,
                                                                     1)  # np.where(j < 0, state_not_done * 0, state_not_done)
                j = j.clip(min=0)

            # State should be all 0s if the episode terminated previously
            state[:, i] = self.state[j] * state_not_done
            next_state[:, i] = self.state[k] * next_not_done

            # If this was the first timestep, make everything previous = 0
            next_not_done *= state_not_done
            state_not_done *= (1. - self.first_timestep[j]).reshape(-1, 1, 1)

        return (
            torch.ByteTensor(state).to(self.device).float(),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.ByteTensor(next_state).to(self.device).float(),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])

        np.save(f"{save_folder}_raw_state.npy", self.raw_state[:self.size])
        np.save(f"{save_folder}_raw_next_state.npy", self.raw_next_state[:self.size])

        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

    def load2(self, save_folder, save_folder2, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

        for i in range(10):
            self.next_state_total[i][:self.size] = np.load(f"{save_folder2}_next_state"+str(i+1)+".npy")[:self.size]

        #self.raw_state[:self.size] = np.load(f"{save_folder}_raw_state.npy")[:self.size]
        #self.raw_next_state[:self.size] = np.load(f"{save_folder}_raw_next_state.npy")[:self.size]

    def concat(self, replay_buffer, num1, num2_, num2):
        self.max_size = num1 + (num2-num2_)
        self.size = num1 + (num2-num2_)

        self.state = np.concatenate((self.state[:num1], replay_buffer.state[num2_:num2]), axis=0)
        self.raw_state = np.concatenate((self.raw_state[:num1], replay_buffer.raw_state[num2_:num2]), axis=0)
        self.action = np.concatenate((self.action[:num1], replay_buffer.action[num2_:num2]), axis=0)
        self.next_state = np.concatenate((self.next_state[:num1], replay_buffer.next_state[num2_:num2]), axis=0)
        self.raw_next_state = np.concatenate((self.raw_next_state[:num1], replay_buffer.raw_next_state[num2_:num2]), axis=0)
        self.reward = np.concatenate((self.reward[:num1], replay_buffer.reward[num2_:num2]), axis=0)
        self.not_done = np.concatenate((self.not_done[:num1], replay_buffer.not_done[num2_:num2]), axis=0)
