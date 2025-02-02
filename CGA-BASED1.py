# for simiplycity we take 50 people(km) and 1(l) UAV-IRS 1(m) Base_Station
import pandas as pd
import numpy as np
import math
import random


P_m_har=None # using loop of size fo Base_station
T_m_har=None # using loop of size fo Base_station
def E_ml_har(P_m_har,T_m_har): # part of eqation number 2  
    return P_m_har*T_m_har   #it should be m*l dimension

#_________________________________________________________________________________________
Dm=0.49 # in Mbits
D_km=0.5 #Mbits
B=B=10 #MHz #bandwidth


P_km_up=None #using loop of size people 
F_km=None #using loop of size people
P_i_up=None #using loop of size people 
g_ml=None #using loop of size fo UAV-IRS
g_km_l=None #using loop of size of UAV-IRS
P_i_up=None #using loop of size people
h_il_up=None #using loop of size people
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



def h_kml_up(g_lm,V_lm_up,g_km_l): #it is inside the equation number 4
    return abs(g_lm*V_lm_up*g_km_l)


def R_kml_up(B,P_km_up,h_kml_up,P_i_up,h_il_up,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ ((P_i_up*h_il_up)+pow(sigma_m,2)) 
    return B*math.log2(1+temp1) 

T_km_com=D_km/F_km #inside the equation 2.Inside data of people_data.csv
T_kml_up=Dm/R_kml_up # equation number 5

h_lm=None #using loop of size of UAV-IRS
h_km_l=None #using loop of size of UAV-IRS
sigma_km=pow(10,-13) #varience of additive white Gaussian noise of population

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



def h_kml_down(h_lm,V_lm_down,h_km_l): #it is inside the equation number 4
    return abs(h_lm*V_lm_down*h_km_l)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/pow(pow(sigma_km),2) # it will return the sigal value which is minimum of all the value for each itaration

def P_m_down(Dm,T_km_com,T_kml_up,Tm,h_ml_worst): #eqation number 25
    temp1=pow(2,((Dm/T_km_com*T_kml_up*Tm)-1)) # it will always between 0 and p_max=10 for now
    return temp1/h_ml_worst

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=h_ml_worst*P_m_down
    return B*math.log2(1+temp1)

T_ml_down=Dm/R_ml_down #eqation number 9


def E_ml_down(P_m_down,T_ml_down): #eqation number 10
    return P_m_down*T_ml_down
#______________________________________________________________________________
Wl=None #take from 5 to 10 like(5,6,7,8,9,10)
V_l_vfly = None # take for loop of 1 to 100
V_lm_hfly=None # take for loop of 1 to 100
delta=2 #for calculat the p_l_b  for now
H =None # take a loop from 1 to 100
Bh=(1-2.2558*pow(10,4)*H) #air density at height H
Ar=0.503 #area of rotor disc
s= 0.05 #Speed of rotor
V_tip= 120 #rotor solidity 120
p_l_b=(delta/8)*Bh*Ar*s*pow(V_tip,3) 
Nr=4 #number of rotor fo UAV-IRS 


def P_l_vfly(Wl,V_l_vfly,P_l_b,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(pow(V_l_vfly,2)+(2*Wl)/temp2)
    return ((Wl/2)*(V_l_vfly+temp3))+Nr*P_l_b

def T_l_vfly(H,V_l_vfly): #eqation number 12
    return H/V_l_vfly

def P_lm_blade(Nr,P_l_b,V_tip,V_lm_hfly): #eqation number 14
    return Nr*P_l_b*(1+((3*pow(V_lm_hfly))/pow(V_tip,2)))

Cd=None #take a loop 10 value bwtween .02 to 1
Af=0.06 #fuselage area in equation 15 0.06

def P_lm_fuselage(Cd,Af,Bh,V_lm_hfly): #eqation number 15
    return (1/2)*Cd*Af*Bh*pow(V_lm_hfly,3)

def P_lm_induced(Nr,Bh,Ar,Wl,V_lm_hfly): #eqation number 16
    return Wl*pow(math.sqrt(pow(Wl,2)/(4*pow(Nr,2)*pow(Bh,2)*pow(Ar,2))+(pow(V_lm_hfly,4)/4))-(pow(V_lm_hfly,2)/2),(1/2))



def P_lm_hfly(P_lm_blade,P_lm_fuselage,P_lm_induced): #eqation number 13
    return P_lm_blade+P_lm_fuselage+P_lm_induced

d_lm_hfly=None # take a loop 0f 0 t0 100

def T_l_hfly(d_lm_hfly,V_lm_hfly): #eqation number 17
    return d_lm_hfly/V_lm_hfly

def P_l_hov(Wl,P_l_b,Nr,Ar,Bh): #eqation number 18
    temp1=Nr*P_l_b
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(2*temp2)
    temp4=pow(Wl,3/2)/temp3
    return temp1+ temp4

D_km=0.5 #Mbits
F_km = None #take a loop of 50 value between 20 to 100

T_km_com=D_km/F_km #inside the equation 2



def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down   #we have to take the max of T_com_km and max of Tup_kml for each round
   

def E_ml_UAV(P_l_vfly,T_l_vfly,P_lm_hfly,T_l_hfly,P_l_hov,T_lm_hov): #energy consumption of the UAV-IRS,
    return P_l_vfly*T_l_vfly+P_lm_hfly*T_l_hfly+P_l_hov*T_lm_hov # eqation number 20

def Fitness(E_ml_har,E_ml_down,E_ml_UAV): #eqation number 30
    return E_ml_har+E_ml_down+E_ml_UAV # it should be m*l dimension





result=Fitness(E_ml_har,E_ml_down,E_ml_UAV) # this could be the size of m*l 2d matrix whcih store
#the mimimum fitness value of for each combination of Base staion and UAV-IRS 
 