import numpy as np

original_next_state = np.load("test_buffer_next_state.npy")

vae_next_state1 = np.load("vae_buffer_next_state1.npy")
vae_next_state2 = np.load("vae_buffer_next_state2.npy")
vae_next_state3 = np.load("vae_buffer_next_state3.npy")
vae_next_state4 = np.load("vae_buffer_next_state4.npy")
vae_next_state5 = np.load("vae_buffer_next_state5.npy")
vae_next_state6 = np.load("vae_buffer_next_state6.npy")
vae_next_state7 = np.load("vae_buffer_next_state7.npy")

original_next_state1 = np.round(original_next_state[5:6],3)

kk1 = np.round(vae_next_state1[5:6],3)
kk2 = np.round(vae_next_state2[5:6],3)
kk3 = np.round(vae_next_state3[5:6],3)
kk4 = np.round(vae_next_state4[5:6],3)
kk5 = np.round(vae_next_state5[5:6],3)
kk6 = np.round(vae_next_state6[5:6],3)
kk7 = np.round(vae_next_state7[5:6],3)

print(original_next_state1)
print(kk1)
print(kk2)
print(kk3)
print(kk4)
print(kk5)
print(kk6)
print(kk7)



# print(np.round(original_next_state,3))
# print("#################3")
# print(np.round(vae_next_state,3))
# print("#################3")
#
# print(np.shape(original_next_state))
# print(np.round(np.mean(original_next_state,axis=0),3))
# print(np.round(np.mean(vae_next_state,axis=0),3))
# print("#############")
# print(np.mean(np.round(original_next_state-vae_next_state,3)))
