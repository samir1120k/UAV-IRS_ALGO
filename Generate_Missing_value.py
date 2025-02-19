import random
import numpy as np
import pandas as pd
from numpy import random
import math
import cmath

# Cd=[]
# Wl=[]
# H=[]
# Angle=[]
# for i in range(50):
#     x=random.uniform(0.02,1)
#     Cd.append(x)
# # print(Cd)

# for i in range(50):
#     x=random.uniform(5,10)
#     Wl.append(x)
# # print(Wl)

# for i in range(50):
#     x=random.uniform(50,100)
#     H.append(x)
# # print(H)

# for i in range(50):
#     x=random.uniform(0,180)
#     Angle.append(x)
# # print(Angle)

# j=1
# Angle=j*Angle
# # # print(Angle)

# Angle=np.exp(Angle)
# # # print(Angle)

# V_lm_down=np.zeros((50,50))

# for i in np.arange(10):
#     for j in np.arange(10):  
#         if (i==j):   
#             V_lm_down[i,j] = Angle[j] 

# # print(V_lm_down)

# h_lm=[]
# for i in range(50):
#     x=random.uniform(1,100)
#     h_lm.append(x)

# h_km_l=[]
# for i in range(50):
#     x=random.uniform(1,100)
#     h_km_l.append(x)
# h_km_l=np.transpose(h_km_l)

# # print(h_km_l)

# h_kml_down=pow((h_lm*h_km_l*V_lm_down),2)

# # print(h_kml_down)

# file = open("file2.txt", "w+")
 
# # Saving the array in a text file
# content = str(h_kml_down)
# file.write(content)
# file.close()
 
# Displaying the contents of the text file
# file = open("file1.txt", "r")
# content = file.read()
 
# print("\nContent in file1.txt:\n", content)
# file.close()


#creat a data frame uisng the parameter and the value are [T_km_com,T_kml_up,Cd,wl,H]
#save in csv file

# df=pd.DataFrame({'Cd':Cd,'Wl':Wl,'H':H})
# df.to_csv('data.csv',index=True)

# df=pd.read_csv("data.csv")
# # print(df.head())
# cd= df['Cd']
# print(cd)


# V_l_vfly=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     V_l_vfly.append(x)
# print(V_l_vfly)

# V_lm_hfly=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     V_lm_hfly.append(x)

# d_lm_hfly=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     d_lm_hfly.append(x)

# f_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     f_up.append(x)

# h_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     h_up.append(x)

# P_km_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     P_km_up.append(x)

# P_i_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     P_i_up.append(x)

# h_il_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     h_il_up.append(x)

# # creating of dataframe of above list

# df1=pd.DataFrame({'h_il_up':h_il_up,'P_i_up':P_i_up,'P_km_up':P_km_up,'h_up':h_up,'f_up':f_up,'d_lm_hfly':d_lm_hfly,'V_l_vfly':V_l_vfly})
# df1.to_csv('data1.csv',index=True)



# T_m_har=[]
# for i in range(20):
#     x=random.uniform(0,1)
#     T_m_har.append(x)

# P_m_har=[]
# for i in range(20):
#     x=random.uniform(0,1)
#     P_m_har.append(x)

# P_m_down=[]
# for i in range(20):
#     x=random.uniform(0,1)
#     P_m_down.append(x)



# # creating of dataframe of above list
# df2=pd.DataFrame({'P_m_har':P_m_har,'T_m_har':T_m_har,'P_m_down':P_m_down,})
# df2.to_csv('BS_data.csv',index=True)

# V_lm_vfly=[]
# for i in range(20):
#     x=random.uniform(0,100)
#     V_lm_vfly.append(x)

# V_lm_hfly=[]
# for i in range(20):
#     x=random.uniform(0,100)
#     V_lm_hfly.append(x)

# D_l_hfly=[]

# for i in range(20):
#     x=random.uniform(0,100)
#     D_l_hfly.append(x)


# df3=pd.DataFrame({'V_lm_vfly':V_lm_vfly,'V_lm_hfly':V_lm_hfly,'D_l_hfly':D_l_hfly})
# df3.to_csv('UAV_data.csv',index=True)


# T_km_com=[]
# for i in range(500):
#     x=random.uniform(0,100)
#     T_km_com.append(x)

# T_km_up=[]
# for i in range(50):
#     x=random.uniform(0,100)
#     T_km_up.append(x)


# df4=pd.DataFrame({'T_km_com':T_km_com,'T_km_up':T_km_up})
# df4.to_csv('people_data.csv',index=True)

# calculating the h_kml_down
# for example we take reflecting surface is 50
# num_of_irs=500
# num_of_people=500
# Angle=np.random.uniform(0,180,(num_of_irs,num_of_people))
# # print(Angle)
# h_l_km=np.random.uniform(0,10,(num_of_irs,num_of_people))
# h_l_m=np.random.uniform(0,10, (num_of_irs,num_of_people))

# print(random.choice(Angle))


# df5=pd.DataFrame(Angle)
# df5.to_csv('Angle.csv',index=False)

# df6=pd.DataFrame(h_l_km)
# df6.to_csv('h_l_km.csv',index=False)

# df5=pd.DataFrame(h_l_m)
# df5.to_csv('h_l_m.csv',index=False)


# def calculate_exp_i_theta(theta):
#   return cmath.exp(1j * theta) 
#  # 1j represents the imaginary unit in Python

# def h_kml_down(Angle,h_l_m,h_l_km):
#   result=[]
#   for i in range(len(Angle)):
#       theta_radians = math.radians(Angle[i])
#       results= calculate_exp_i_theta(theta_radians)
#       result.append(results)

#   diagonal=np.diag(result)
#   h_l_km.transpose()

#   a=np.dot(h_l_km,diagonal)
#   b=np.dot(a,h_l_m)
#   final=abs(b)
#   return final**2

# final=h_kml_down(Angle[0,:],h_l_m[0,:],h_l_km[0,:])

# print(final)

# f_km_up=np.random.uniform(0.1,1,500)

# df5=pd.DataFrame(f_km_up)
# df5.to_csv('f_km.csv',index=False)
# g_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50
# print(g_l_m_df.shape)
# print(random.randint(10))
# f_km1=pd.read_csv(r'f_km.csv')
# f_km=f_km1['0']
# l=2
# f_km2=f_km[l*50:(l+1)*50]
# print(f_km2)
# plt.rcParams["font.size"] = "20"
# # plt.rcParams['figure.figsize'] = [5, 6]
# # plt.plot(x, y, color='red', linestyle='dashed', linewidth = 0.5, markerfacecolor='blue', markersize=12)
# plt.plot(T_m_range, fitness_sums_D_m, label = "HC-A")
# plt.plot(T_m_range, fitness_sums_p_max, label = "C2GA")
# plt.xlabel('Data size',size=20)
# plt.ylabel('Energy',size=22)
# # giving a title to my graph
# # plt.title('Reward')
# plt.legend()
# plt.savefig("Energy vs Data size.pdf", format="pdf", bbox_inches="tight", dpi=800)
# # plt.legend()
# plt.show()


# h_l_km=np.random.uniform(0,10,500)





# df5=pd.DataFrame(h_l_km)
# df5.to_csv('P_km_up.csv',index=False)

# print(random.randint(0,50))
# print(random.normal(loc=0, scale=1, size=(1))[0])