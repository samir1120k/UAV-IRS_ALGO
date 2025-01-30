import numpy as np
import random
import math
import pandas as pd

df=pd.read_csv("data.csv")


#define required variable
Cd= df['Cd'] #drag coefficient in equation 15
Af=0.06 #fuselage area in equation 15 0.06
Wl=df['Wl'] #weight of UAV-IRS 5-10
Nr=4 #number of rotor fo UAV-IRS 
delta=None #for calculat the p_l_b
V_tip= 120 #rotor solidity 120
H=df['H'] #height of UAV-IRS 
Ar=0.503 #area of rotor disc
s= 0.05 #Speed of rotor
Bh=(1-2.2558*pow(10,4)*H) #air density at height H
p_l_b=(delta/8)*Bh*Ar*s*pow(V_tip,3) 
sigma_km=pow(10,-13) #varience of additive white Gaussian noise of population
sigma_m=pow(10,-13) #varience of additive white Gaussian noise of base_station
B=10 #MHz #bandwidth
Dm=0.49 # Mbits
D_km=0.5 #Mbits
F_km = 0.5 #Mbits Assuming
Tm=10 #sec




def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=h_ml_worst*P_m_down
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/pow(pow(sigma_km),2) # W for minimum selection we put the value of 1

# def h_kml_down(h_km_l,V_lm_down,h_lm):
#     temp=[]
#     for i in range(len(h_lm)):
#         temp.append(h_km_l*V_lm_down*h_lm)
#     temp2=min(temp)
    return temp2

# def h_kml_down(h_km_l,V_lm_down,h_lm): #it is inside the equation number 8
#     return abs(h_km_l*V_lm_down*h_lm)

# calculate and store in the file
# open file and import the value 
file = open("file1.txt", "r")
h_kml_down = file.read()

file1 = open("file2.txt", "r") #inside the equation 4
h_kml_up = file1.read()

T_ml_down=Dm/R_ml_down #eqation number 9


def P_m_down(Dm,T_km_com,T_kml_up,Tm,h_ml_worst): #eqation number 25
    temp1=pow(2,((Dm/T_km_com*T_kml_up*Tm)-1))
    return temp1/h_ml_worst
    #condition have to be satisfied of eqation number 26
    #0<P_down_m<=P_max_m
    #still some value is not defined in eqation number 25
    #T_com_km_str
    #T_up_kml_str
    #Tm

V_l_vfly = None #m/s express as follow[35] ,inside the dataframe
V_lm_hfly= None # forward flight to the HP is derived using teh axial momentum theory express as follow[38]
                # defined inside the dataframe

def P_lm_blade(Nr,P_l_b,V_tip,V_lm_hfly): #eqation number 14
    return Nr*P_l_b*(1+((3*pow(V_lm_hfly))/pow(V_tip,2)))

def P_lm_fuselage(Cd,Af,Bh,V_lm_hfly): #eqation number 15
    return (1/2)*Cd*Af*Bh*pow(V_lm_hfly,3)

def P_lm_induced(Nr,Bh,Ar,Wl,V_lm_hfly): #eqation number 16
    return Wl*pow(math.sqrt(pow(Wl,2)/(4*pow(Nr,2)*pow(Bh,2)*pow(Ar,2))+(pow(V_lm_hfly,4)/4))-(pow(V_lm_hfly,2)/2),(1/2))


d_lm_hfly = None #horizontal distance between the UAV-IRS and the HP
                 #defined inside the dataframe

T_km_com=D_km/F_km #inside the equation 2

def R_kml_up(B,P_km_up,h_kml_up,P_i_up,h_il_up,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ ((P_i_up*h_il_up)+pow(sigma_m,2))  #Need attention on this equation
    return B*math.log2(1+temp1)  #defined in the dataframe

T_kml_up=Dm/R_kml_up # equation number 5



def P_l_vfly(Wl,V_l_vfly,P_l_b,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(pow(V_l_vfly,2)+(2*Wl)/temp2)
    return ((Wl/2)*(V_l_vfly+temp3))+Nr*P_l_b

def T_l_vfly(H,V_l_vfly): #eqation number 12
    return H/V_l_vfly

def P_lm_hfly(P_lm_blade,P_lm_fuselage,P_lm_induced): #eqation number 13
    return P_lm_blade+P_lm_fuselage+P_lm_induced

def T_l_hfly(d_lm_hfly,V_lm_hfly): #eqation number 17
    return d_lm_hfly/V_lm_hfly

def P_l_hov(Wl,P_l_b,Nr,Ar,Bh): #eqation number 18
    temp1=Nr*P_l_b
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(2*temp2)
    temp4=pow(Wl,3/2)/temp3
    return temp1+ temp4

def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down
    #we have to take the max{T_com_km,Tup_kml}

def E_ml_UAV(P_l_vfly,T_l_vfly,P_lm_hfly,T_l_hfly,P_l_hov,T_lm_hov): #energy consumption of the UAV-IRS,
    return P_l_vfly*T_l_vfly+P_lm_hfly*T_l_hfly+P_l_hov*T_lm_hov # eqation number 20

def E_ml_down(P_m_down,T_ml_down): #eqation number 10
    return P_m_down*T_ml_down

def E_ml_har(P_m_har,T_m_ahr): # part of eqation number 2
    return P_m_har*T_m_ahr

def Fitness(E_ml_har,E_ml_down,E_ml_UAV): #eqation number 30
    return E_ml_har+E_ml_down+E_ml_UAV

#above all the function are define now we have to define the fitness function

#genetic algorithm to optimize the UAV-IRS assignment
