import pandas as pd
import math
import numpy as np
import random
Dm=0.49 # in Mbits
df= pd.read_csv(r'data1.csv')
Dm=0.49 # in Mbits
D_km=0.5 #Mbits
B=B=10 #MHz #bandwidth


P_km_up=df['P_km_up'] #using loop of size people 
F_km=df['f_up'] #using loop of size people
P_i_up=df['P_i_up'] #using loop of size people 
g_ml=df['V_l_vfly'] #using loop of size fo UAV-IRS
g_km_l=df['d_lm_hfly'] #using loop of size of UAV-IRS
h_il_up=df['h_il_up'] #using loop of size people
sigma_m=pow(10,-13) #varience of additive white Gaussian noise of base_station
Tm=10 #sec

Angle=[]
for i in range(10):
    x=random.uniform(0,180)
    Angle.append(x)
j=1
Angle=j*Angle
Angle=np.exp(Angle)
V_lm_up=np.zeros((10,10))

for i in np.arange(10):
    for j in np.arange(10):  
        if (i==j):   
            V_lm_up[i,j] = Angle[j] 

Angle1=[]
for i in range(10):
    x=random.uniform(0,180)
    Angle1.append(x)
j1=1
Angle1=j1*Angle1
Angle1=np.exp(Angle1)
V_lm_down=np.zeros((10,10))

for i in np.arange(10):
    for j in np.arange(10):  
        if (i==j):   
            V_lm_down[i,j] = Angle1[j1] 

result=[]

for i in F_km:
    for j in P_km_up:
        for k in P_i_up:
            for l in g_ml:
                for m in g_km_l:
                    for n in h_il_up:    
                        temp3=random.choice(V_lm_up[1:])   
                        temp4=random.choice(V_lm_down[1:])
                        def h_kml_up(g_lm,V_lm_up,g_km_l): #it is inside the equation number 4
                            return abs(g_lm*V_lm_up*g_km_l)
                        def R_kml_up(B,P_km_up,h_kml_up,P_i_up,h_il_up,sigma_m): #eqation number 4
                            temp1=(P_km_up*h_kml_up)/ ((P_i_up*h_il_up)+pow(sigma_m,2)) 
                            return B*math.log2(1+1) 
                        temp2=h_kml_up(l,temp3,m)
                        temp=R_kml_up(B,j,temp2,k,n,sigma_m)

                        T_km_com=D_km/i #inside the equation 2.Inside data of people_data.csv
                        T_kml_up=Dm/temp # equation number 5

                        h_lm=None #using loop of size of UAV-IRS
                        h_km_l=None #using loop of size of UAV-IRS
                        sigma_km=pow(10,-13) #varience of additive white Gaussian noise of population


                        def h_kml_down(h_lm,V_lm_down,h_km_l): #it is inside the equation number 4
                            return abs(h_lm*V_lm_down*h_km_l)

                        def h_ml_worst(h_kml_down,sigma_m): #eqation number 8
                            return h_kml_down/pow(sigma_m,2) # it will return the sigal value which is minimum of all the value for each itaration
                        temp5=h_ml_worst(h_kml_down,sigma_km)
                        def P_m_down(Dm,T_km_com,T_kml_up,Tm,h_ml_worst): #eqation number 25
                            temp1=pow(2,((Dm/T_km_com*T_kml_up*Tm)-1)) # it will always between 0 and p_max=10 for now
                            return temp1/temp5
                        
                        result =P_m_down(Dm,T_km_com,T_kml_up,Tm,h_ml_worst)

print(result)