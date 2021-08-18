import numpy as np
import matplotlib.pyplot as plt

bcq_speed1 =np.load('./results/BCQ_original2_final_00_speeds.npy', allow_pickle=True)
bcq_speed2 =np.load('./results/BCQ_original2_final_01_speeds.npy', allow_pickle=True)
bcq_speed3 =np.load('./results/BCQ_original2_final_02_speeds.npy', allow_pickle=True)
bcq_speed =(bcq_speed1+bcq_speed2+bcq_speed3)/3
bcq_trackpos1 =np.load('./results/BCQ_original2_final_00_tracpos.npy', allow_pickle=True)
bcq_trackpos2 =np.load('./results/BCQ_original2_final_01_tracpos.npy', allow_pickle=True)
bcq_trackpos3=np.load('./results/BCQ_original2_final_02_tracpos.npy', allow_pickle=True)
bcq_trackpos =(bcq_trackpos1+bcq_trackpos2+bcq_trackpos3)/3

bcq_s_var1 =np.load('./results/BCQ_original2_final_00_var_s.npy', allow_pickle=True)
bcq_s_var2 =np.load('./results/BCQ_original2_final_01_var_s.npy', allow_pickle=True)
bcq_s_var3 =np.load('./results/BCQ_original2_final_02_var_s.npy', allow_pickle=True)
bcq_s_var =(bcq_s_var1+bcq_s_var2+bcq_s_var3)/3
bcq_t_var1 =np.load('./results/BCQ_original2_final_00_var_t.npy', allow_pickle=True)
bcq_t_var2 =np.load('./results/BCQ_original2_final_01_var_t.npy', allow_pickle=True)
bcq_t_var3 =np.load('./results/BCQ_original2_final_02_var_t.npy', allow_pickle=True)
bcq_t_var =(bcq_t_var1+bcq_t_var2+bcq_t_var3)/3

ddpg_speed1 = np.load('./results/DDPG_final_0_speeds.npy', allow_pickle=True)
ddpg_speed2 = np.load('./results/DDPG_final_1_speeds.npy', allow_pickle=True)
ddpg_speed3 = np.load('./results/DDPG_final_2_speeds.npy', allow_pickle=True)
ddpg_speed =(ddpg_speed1+ddpg_speed2+ddpg_speed3)/3
ddpg_trackpos1 = np.load('./results/DDPG_final_0_tracpos.npy', allow_pickle=True)
ddpg_trackpos2 = np.load('./results/DDPG_final_1_tracpos.npy', allow_pickle=True)
ddpg_trackpos3 = np.load('./results/DDPG_final_2_tracpos.npy', allow_pickle=True)
ddpg_trackpos =(ddpg_trackpos1+ddpg_trackpos2+ddpg_trackpos3)/3

ddpg_s_var1 = np.load('./results/DDPG_final_0_var_s.npy', allow_pickle=True)
ddpg_s_var2 = np.load('./results/DDPG_final_1_var_s.npy', allow_pickle=True)
ddpg_s_var3 = np.load('./results/DDPG_final_2_var_s.npy', allow_pickle=True)
ddpg_s_var=(ddpg_s_var1+ddpg_s_var2+ddpg_s_var3)/3
ddpg_t_var1 = np.load('./results/DDPG_final_0_var_t.npy', allow_pickle=True)
ddpg_t_var2 = np.load('./results/DDPG_final_1_var_t.npy', allow_pickle=True)
ddpg_t_var3 = np.load('./results/DDPG_final_2_var_t.npy', allow_pickle=True)
ddpg_t_var=(ddpg_t_var1+ddpg_t_var2+ddpg_t_var3)/3


max_original_speed = []
min_original_speed = []
max_original_t = []
min_original_t = []
max_ddpg_speed = []
min_ddpg_speed = []
max_ddpg_t = []
min_ddpg_t = []
max_bcq_s_var =[]
min_bcq_s_var =[]
max_ddpg_s_var = []
min_ddpg_s_var = []
max_bcq_t_var = []
min_bcq_t_var = []
max_ddpg_t_var = []
min_ddpg_t_var = []

for i in range (len(bcq_speed)):
    max_original_speed.append(max([bcq_speed1[i],bcq_speed2[i],bcq_speed3[i]]))
    min_original_speed.append(min([bcq_speed1[i], bcq_speed2[i], bcq_speed3[i]]))
    max_ddpg_speed.append(max([ddpg_speed1[i],ddpg_speed2[i],ddpg_speed3[i]]))
    min_ddpg_speed.append(min([ddpg_speed1[i],ddpg_speed2[i],ddpg_speed3[i]]))

    max_original_t.append(max([bcq_trackpos1[i],bcq_trackpos2[i],bcq_trackpos3[i]]))
    min_original_t.append(min([bcq_trackpos1[i],bcq_trackpos2[i],bcq_trackpos3[i]]))
    max_ddpg_t.append(max([ddpg_trackpos1[i],ddpg_trackpos2[i],ddpg_trackpos3[i]]))
    min_ddpg_t.append(min([ddpg_trackpos1[i],ddpg_trackpos2[i],ddpg_trackpos3[i]]))

    max_bcq_s_var.append(max([bcq_s_var1[i],bcq_s_var2[i],bcq_s_var3[i]]))
    min_bcq_s_var.append(min([bcq_s_var1[i], bcq_s_var2[i], bcq_s_var3[i]]))
    max_ddpg_s_var.append(max([ddpg_s_var1[i],ddpg_s_var2[i],ddpg_s_var3[i]]))
    min_ddpg_s_var.append(min([ddpg_s_var1[i],ddpg_s_var2[i],ddpg_s_var3[i]]))

    max_bcq_t_var.append(max([bcq_t_var1[i],bcq_t_var2[i],bcq_t_var3[i]]))
    min_bcq_t_var.append(min([bcq_t_var1[i],bcq_t_var2[i],bcq_t_var3[i]]))
    max_ddpg_t_var.append(max([ddpg_t_var1[i],ddpg_t_var2[i],ddpg_t_var3[i]]))
    min_ddpg_t_var.append(min([ddpg_t_var1[i],ddpg_t_var2[i],ddpg_t_var3[i]]))


plt.subplot(211)
plt.plot(list(range(len(bcq_speed))), bcq_s_var, c="g", lw=2, ls="-",label='BCQ')
plt.plot(list(range(len(ddpg_speed))), ddpg_s_var, c="b", lw=2, ls="-",label='DDPG')

plt.legend(loc='lower right')
plt.ylabel("speed")
plt.xlabel("episode")

plt.fill_between(list(range(len(bcq_speed))), max_bcq_s_var, min_bcq_s_var,alpha =0.15,color ='g')
plt.fill_between(list(range(len(bcq_speed))), max_ddpg_s_var, min_ddpg_s_var,alpha =0.15,color ='b')

plt.subplot(212)
plt.plot(list(range(len(bcq_trackpos))), bcq_t_var, c="g", lw=2, ls="-",label='BCQ')
plt.plot(list(range(len(ddpg_trackpos))), ddpg_t_var, c="b", lw=2, ls="-",label='DDPG')
plt.ylabel("trackpos")
plt.xlabel("episode")
plt.legend(loc='lower right')

plt.fill_between(list(range(len(bcq_speed))), max_bcq_t_var, min_bcq_t_var,alpha =0.15,color ='g')
plt.fill_between(list(range(len(bcq_speed))), max_ddpg_t_var, min_ddpg_t_var,alpha =0.15,color ='b')
#
# plt.subplot(121)
# plt.plot(list(range(len(sp_s_var))), sp_s_var, c="b", lw=2, ls="--",label='BCQ_speed')
# plt.plot(list(range(len(pos_t_var))), pos_t_var, c="r", lw=2, ls="--",label='BCQ_trackpos')
#
# plt.subplot(122)
# plt.plot(list(range(len(sp_t_var))), sp_t_var, c="b", lw=2, ls="--",label='BCQ_speed')
# plt.plot(list(range(len(pos_s_var))), pos_s_var, c="r", lw=2, ls="--",label='BCQ_trackpos')

plt.tight_layout()
plt.gcf().set_size_inches(5.5, 6)
plt.savefig('variance.png', dpi =500)
plt.show()
