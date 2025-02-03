import pandas as pd
import random 
import math

base=pd.read_csv(r'BS_data.csv')
uav=pd.read_csv(r'UAV_data.csv')
people=pd.read_csv(r'people_data.csv')


Wl=[4,5,6,7,8,9,10]
H=base['H']
P_m_har=base['P_m_har']
T_m_har=base['T_m_har']
P_m_down=base['P_m_down']
T_ml_down=base['T_ml_down']
T_km_com=people['T_km_com']
T_km_up=people['T_km_up']
V_lm_vfly=uav['V_lm_vfly']
V_lm_hfly=uav['V_lm_hfly']
D_l_hfly=uav['D_l_hfly']

delta=2 #for calculat the p_l_b  for now
Ar=0.503 #area of rotor disc
s= 0.05 #Speed of rotor
Nr=4 #number of rotor fo UAV-IRS
V_tip= 120 #rotor solidity 120
Cd=0.5 #take a loop 10 value bwtween .02 to 1
Af=0.06 #fuselage area in equation 15 0.06


def Fitness(E_ml_har,E_ml_down,E_ml_UAV): #eqation number 30
    return E_ml_har+E_ml_down+E_ml_UAV # it should be m*l dimension

def E_ml_UAV(P_l_vfly,T_l_vfly,P_lm_hfly,T_l_hfly,P_l_hov,T_lm_hov): #energy consumption of the UAV-IRS,
    return P_l_vfly*T_l_vfly+P_lm_hfly*T_l_hfly+P_l_hov*T_lm_hov # eqation number 20

def P_l_vfly(Wl,V_l_vfly,P_l_b,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(pow(V_l_vfly,2)+(2*Wl)/temp2)
    return ((Wl/2)*(V_l_vfly+temp3))+Nr*P_l_b

def P_lm_hfly(P_lm_blade,P_lm_fuselage,P_lm_induced): #eqation number 13
    return P_lm_blade+P_lm_fuselage+P_lm_induced

def P_lm_blade(Nr,P_l_b,V_tip,V_lm_hfly): #eqation number 14
    return Nr*P_l_b*(1+((3*pow(V_lm_hfly,2))/pow(V_tip,2)))

def P_lm_fuselage(Cd,Af,Bh,V_lm_hfly): #eqation number 15
    return (1/2)*Cd*Af*Bh*pow(V_lm_hfly,3)

def P_lm_induced(Nr,Bh,Ar,Wl,V_lm_hfly): #eqation number 16
    return Wl*pow(math.sqrt(pow(Wl,2)/(4*pow(Nr,2)*pow(Bh,2)*pow(Ar,2))+(pow(V_lm_hfly,4)/4))-(pow(V_lm_hfly,2)/2),(1/2))

def P_l_hov(Wl,P_l_b,Nr,Ar,Bh): #eqation number 18
    temp1=Nr*P_l_b
    temp2=abs(2*(Nr*Bh*Ar))
    temp3=math.sqrt(temp2)
    temp4=pow(Wl,3/2)/temp3
    return temp1+ temp4

def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down   #we have to take the max of T_com_km and max 

parant1=None
result=[]

for i in range(2):
    H_value=random.choice(H)
    Wl_value=random.choice(Wl)
    P_m_down_value=random.choice(P_m_down)
    P_m_har_value=random.choice(P_m_har)
    T_m_har_value=random.choice(T_m_har)
    T_ml_down_value=random.choice(T_ml_down)
    T_km_com_value=random.choice(T_km_com)
    T_km_up_value=random.choice(T_km_up)
    V_lm_vfly_value=random.choice(V_lm_vfly)
    V_lm_hfly_value=random.choice(V_lm_vfly)
    D_l_hfly_value=random.choice(D_l_hfly)

    Bh=(1-2.2558*pow(10,4)*H_value) #air density at height H
    p_l_b=(delta/8)*Bh*Ar*s*pow(V_tip,3) 
   
    P_lm_blade_value=P_lm_blade(Nr,p_l_b,V_tip,V_lm_hfly_value)
    P_lm_fuselage_value=P_lm_fuselage(Cd,Af,Bh,V_lm_hfly_value)
    P_lm_induced_value=P_lm_induced(Nr,Bh,Ar,Wl_value,V_lm_hfly_value)

    T_l_vfly_value=H/V_lm_vfly_value
    T_l_hfly_value=D_l_hfly_value*V_lm_hfly_value
    E_ml_har_value=P_m_har_value*T_m_har_value
    E_ml_down_value=P_m_down_value*T_ml_down_value
    T_lm_hov_value=T_lm_hov(T_km_com_value,T_km_up_value,T_ml_down_value)
    P_l_hov_value=P_l_hov(Wl_value,p_l_b,Nr,Ar,Bh)
    P_l_vfly_value=P_l_vfly(Wl_value,V_lm_vfly_value,p_l_b,Nr,Ar,Bh)
    P_lm_hfly_value=P_lm_hfly(P_lm_blade_value,P_lm_fuselage_value,P_lm_induced_value)
    E_ml_UAV_value=E_ml_UAV(P_l_vfly_value,T_l_vfly_value,P_lm_hfly_value,T_l_hfly_value,P_l_hov_value,T_lm_hov_value)

    results=Fitness(E_ml_har_value,E_ml_down_value,E_ml_UAV_value)
    result.append(results)



print(result)