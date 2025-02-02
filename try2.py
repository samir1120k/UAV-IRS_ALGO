import pandas as pd
import math
import numpy as np
import random

Dm = 0.49 # in Mbits
df = pd.read_csv(r'data1.csv')
D_km = 0.5 # Mbits
B = 10 # MHz bandwidth

P_km_up = df['P_km_up'] # using loop of size people
F_km = df['f_up'] # using loop of size people
P_i_up = df['P_i_up'] # using loop of size people
g_ml = df['V_l_vfly'] # using loop of size for UAV-IRS
g_km_l = df['d_lm_hfly'] # using loop of size of UAV-IRS
h_il_up = df['h_il_up'] # using loop of size people
sigma_m = pow(10, -13) # variance of additive white Gaussian noise of base station
Tm = 10 # sec

Angle = [random.uniform(0, 180) for _ in range(10)]
Angle = np.exp(Angle)
V_lm_up = np.diag(Angle)

Angle1 = [random.uniform(0, 180) for _ in range(10)]
Angle1 = np.exp(Angle1)
V_lm_down = np.diag(Angle1)

# Precompute flattened versions of V_lm_up and V_lm_down
V_lm_up_flatten = V_lm_up.flatten()
V_lm_down_flatten = V_lm_down.flatten()

def h_kml_up(g_lm, V_lm_up, g_km_l):
    return abs(g_lm * V_lm_up * g_km_l)

def R_kml_up(B, P_km_up, h_kml_up, P_i_up, h_il_up, sigma_m):
    temp1 = (P_km_up * h_kml_up) / ((P_i_up * h_il_up) + sigma_m**2)
    return B * math.log2(1 + temp1)

def h_kml_down(h_lm, V_lm_down, h_km_l):
    return abs(h_lm * V_lm_down * h_km_l)

def h_ml_worst(h_kml_down, sigma_km):
    return h_kml_down / sigma_km**2

def P_m_down(Dm, T_km_com, T_kml_up, Tm, h_ml_worst):
    if T_km_com * T_kml_up * Tm == 0 or h_ml_worst == 0:
        return float('inf')  # to avoid division by zero
    log_temp1 = math.log2(Dm) - math.log2(T_km_com * T_kml_up * Tm)
    temp1 = pow(2, log_temp1)
    return temp1 / h_ml_worst

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=h_ml_worst*P_m_down
    return B*math.log2(1+temp1)

T_ml_down=Dm/R_ml_down #eqation number 9




result = float('inf')
result1=float('inf')
best_values = {}

for i in F_km:
    for j in P_km_up:
        for k in P_i_up:
            for l in g_ml:
                for m in g_km_l:
                    for n in h_il_up:
                        temp3 = random.choice(V_lm_up_flatten)
                        temp4 = random.choice(V_lm_down_flatten)
                        h_kml_up_val = h_kml_up(l, temp3, m)
                        R_kml_up_val = R_kml_up(B, j, h_kml_up_val, k, n, sigma_m)

                        if R_kml_up_val != 0:
                            T_km_com = D_km / i
                            T_kml_up = Dm / R_kml_up_val

                            h_kml_down_val = h_kml_down(l, temp4, m)
                            h_ml_worst_val = h_ml_worst(h_kml_down_val, sigma_m)

                            P_m_down_val = P_m_down(Dm, T_km_com, T_kml_up, Tm, h_ml_worst_val)

                            if P_m_down_val < result:
                                result = P_m_down_val
                                best_values = {
                                    "i": i,
                                    "j": j,
                                    "k": k,
                                    "l": l,
                                    "m": m,
                                    "n": n
                                }
                            T_ml_down_val=Dm/R_ml_down
                            if T_ml_down_val < result1:
                                result1 =T_ml_down_val
                            
                            
                            

print(f"Minimum P_m_down: {result}")
print(f"Minimum T_m_down: {result1}")
print(f"Best values: {best_values}")
