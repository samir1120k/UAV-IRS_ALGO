import random
import numpy as np
import pandas as pd
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
# # print(Angle)

# Angle=np.exp(Angle)
# # print(Angle)

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


V_l_vfly=[]
for i in range(50):
    x=random.uniform(0,100)
    V_l_vfly.append(x)
print(V_l_vfly)

V_lm_hfly=[]
for i in range(50):
    x=random.uniform(0,100)
    V_lm_hfly.append(x)

d_lm_hfly=[]
for i in range(50):
    x=random.uniform(0,100)
    d_lm_hfly.append(x)

f_up=[]
for i in range(50):
    x=random.uniform(0,100)
    f_up.append(x)

h_up=[]
for i in range(50):
    x=random.uniform(0,100)
    h_up.append(x)

P_km_up=[]
for i in range(50):
    x=random.uniform(0,100)
    P_km_up.append(x)

P_i_up=[]
for i in range(50):
    x=random.uniform(0,100)
    P_i_up.append(x)

h_il_up=[]
for i in range(50):
    x=random.uniform(0,100)
    h_il_up.append(x)

# creating of dataframe of above list

df1=pd.DataFrame({'h_il_up':h_il_up,'P_i_up':P_i_up,'P_km_up':P_km_up,'h_up':h_up,'f_up':f_up,'d_lm_hfly':d_lm_hfly,'V_l_vfly':V_l_vfly})
df1.to_csv('data1.csv',index=True)





