import pandas as pd
import numpy as np
import math
import random

# Constants
Dm = 0.49  # in Mbits
D_km = 0.5  # in Mbits
B = 10  # MHz, bandwidth
sigma_m = pow(10, -13)  # variance of additive white Gaussian noise of base station
sigma_km = pow(10, -13)  # variance of additive white Gaussian noise of population
Tm = 10  # sec
Ar = 0.503  # area of rotor disc
s = 0.05  # speed of rotor
V_tip = 120  # rotor solidity
Nr = 4  # number of rotors for UAV-IRS
Af = 0.06  # fuselage area
delta = 2  # for calculating P_l_b
Cd_values = np.linspace(0.02, 1, 10)  # drag coefficients
Wl_values = np.linspace(5, 10, 10)  # weight values
V_l_vfly_values = np.linspace(1, 100, 100)  # vertical flight speed values
V_lm_hfly_values = np.linspace(1, 100, 100)  # horizontal flight speed values
H_values = np.linspace(1, 100, 100)  # height values
d_lm_hfly_values = np.linspace(0, 100, 100)  # horizontal distance values
F_km_values = np.linspace(20, 100, 50)  # computation capacities

# Functions
def E_ml_har(P_m_har, T_m_har): 
    """Calculates energy harvesting matrix"""
    # Return dimension: m*l
    return P_m_har * T_m_har

def h_kml_up(g_lm, V_lm_up, g_km_l):
    """Calculates uplink channel gain matrix"""
    # Return dimension: m*l
    return abs(g_lm * V_lm_up * g_km_l)

def R_kml_up(B, P_km_up, h_kml_up, P_i_up, h_il_up, sigma_m):
    """Calculates uplink data rate matrix"""
    # Return dimension: m*l
    temp1 = (P_km_up * h_kml_up) / ((P_i_up * h_il_up) + pow(sigma_m, 2))
    return B * np.log2(1 + temp1)

def T_km_com(D_km, F_km):
    """Calculates computation time for users"""
    # Return dimension: k*m
    return D_km / F_km

def T_kml_up(Dm, R_kml_up):
    """Calculates uplink transmission time matrix"""
    # Return dimension: m*l
    return Dm / R_kml_up

def h_kml_down(h_lm, V_lm_down, h_km_l):
    """Calculates downlink channel gain matrix"""
    # Return dimension: m*l
    return abs(h_lm * V_lm_down * h_km_l)

def h_ml_worst(h_kml_down, sigma_km):
    """Calculates worst-case downlink channel gain"""
    # Return dimension: scalar
    return h_kml_down / pow(sigma_km, 2)

def P_m_down(Dm, T_km_com, T_kml_up, Tm, h_ml_worst):
    """Calculates downlink transmission power matrix"""
    # Return dimension: m*l
    temp1 = pow(2, ((Dm / (T_km_com * T_kml_up * Tm)) - 1))
    return temp1 / h_ml_worst

def R_ml_down(B, P_m_down, h_ml_worst):
    """Calculates downlink data rate matrix"""
    # Return dimension: m*l
    temp1 = h_ml_worst * P_m_down
    return B * np.log2(1 + temp1)

def T_ml_down(Dm, R_ml_down):
    """Calculates downlink transmission time matrix"""
    # Return dimension: m*l
    return Dm / R_ml_down

def E_ml_down(P_m_down, T_ml_down):
    """Calculates energy consumption in the downlink"""
    # Return dimension: m*l
    return P_m_down * T_ml_down

def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
    """Calculates power consumption during vertical flight"""
    # Return dimension: m*l
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(pow(V_l_vfly, 2) + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def T_l_vfly(H, V_l_vfly):
    """Calculates time for vertical flight"""
    # Return dimension: m*l
    return H / V_l_vfly

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
    """Calculates power consumption by blades during horizontal flight"""
    # Return dimension: m*l
    return Nr * P_l_b * (1 + ((3 * pow(V_lm_hfly, 2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
    """Calculates power consumption by fuselage during horizontal flight"""
    # Return dimension: m*l
    return (1 / 2) * Cd * Af * Bh * pow(V_lm_hfly, 3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
    """Calculates induced power consumption during horizontal flight"""
    # Return dimension: m*l
    return Wl * np.sqrt(pow(Wl, 2) / (4 * pow(Nr, 2) * pow(Bh, 2) * pow(Ar, 2)) + (pow(V_lm_hfly, 4) / 4) - (pow(V_lm_hfly, 2) / 2))

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
    """Calculates total power consumption during horizontal flight"""
    # Return dimension: m*l
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def T_l_hfly(d_lm_hfly, V_lm_hfly):
    """Calculates time for horizontal flight"""
    # Return dimension: m*l
    return d_lm_hfly / V_lm_hfly

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
    """Calculates power consumption during hovering"""
    # Return dimension: m*l
    temp1 = Nr * P_l_b
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(2 * temp2)
    temp4 = pow(Wl, 3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
    """Calculates hovering time"""
    # Return dimension: m*l
    return T_km_com + T_kml_up + T_ml_down

def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    """Calculates total energy consumption of the UAV-IRS"""
    # Return dimension: m*l
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    """Calculates total fitness value (energy consumption)"""
    # Return dimension: m*l
    return E_ml_har + E_ml_down + E_ml_UAV

# Initialize parameters
P_m_har_values = np.linspace(1, 100, 10)
T_m_har_values = np.linspace(1, 100, 10)
P_km_up_values = np.linspace(1, 100, 50)
P_i_up_values = np.linspace(1, 100, 50)
g_ml_values = np.linspace(1, 100, 10)
g_km_l_values = np.linspace(1, 100, 10)
h_il_up_values = np.linspace(1, 100, 50)
h_lm_values = np.linspace(1, 100, 10)
h_km_l_values = np.linspace(1, 100, 10)

# Initialize matrices
results_matrix = np.zeros((10, 10))

# Main loop to calculate fitness for each combination of base station and UAV-IRS
for i, P_m_har in enumerate(P_m_har_values):
    for j, T_m_har in enumerate(T_m_har_values):
        for k, P_km_up in enumerate(P_km_up_values):
            for l, F_km in enumerate(F_km_values):
                for m, P_i_up in enumerate(P_i_up_values):
                    for n, g_ml in enumerate(g_ml_values):
                        for o, g_km_l in enumerate(g_km_l_values):
                            for p, h_il_up in enumerate(h_il_up_values):
                                for q, h_lm in enumerate(h_lm_values):
                                    for r, h_km_l in enumerate(h_km_l_values):
                                        
                                        # Calculation of V_lm_up
                                        V_lm_up = np.zeros((10, 10))
                                        Angle = np.exp(np.radians(random.uniform(0, 180)))
                                        for x in np.arange(10):
                                            V_lm_up[x, x] = Angle
                                            
                                        # Calculation of V_lm_down
                                        V_lm_down = np.zeros((10, 10))
                                        Angle1 = np.exp(np.radians(random.uniform(0, 180)))
                                        for y in np.arange(10):
                                            V_lm_down[y, y] = Angle1
                                        
                                        # Uplink and downlink calculations
                                        h_kml_up_value = h_kml_up(g_ml, V_lm_up, g_km_l)
                                        R_kml_up_value = R_kml_up(B, P_km_up, h_kml_up_value, P_i_up, h_il_up, sigma_m)
                                        T_km_com_value = T_km_com(D_km, F_km)
                                        T_kml_up_value = T_kml_up(Dm, R_kml_up_value)
                                        h_kml_down_value = h_kml_down(h_lm, V_lm_down, h_km_l)
                                        h_ml_worst_value = h_ml_worst(h_kml_down_value, sigma_km)
                                        P_m_down_value = P_m_down(Dm, T_km_com_value, T_kml_up_value, Tm, h_ml_worst_value)
                                        R_ml_down_value = R_ml_down(B, P_m_down_value, h_ml_worst_value)
                                        T_ml_down_value = T_ml_down(Dm, R_ml_down_value)
                                        E_ml_har_value = E_ml_har(P_m_har, T_m_har)
                                        E_ml_down_value = E_ml_down(P_m_down_value, T_ml_down_value)
                                        
                                        # UAV energy calculations
                                        for Wl in Wl_values:
                                            for V_l_vfly in V_l_vfly_values:
                                                Bh = (1 - 2.2558 * pow(10, -4) * random.choice(H_values))
                                                P_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)
                                                P_l_vfly_value = P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh)
                                                T_l_vfly_value = T_l_vfly(random.choice(H_values), V_l_vfly)
                                                for V_lm_hfly in V_lm_hfly_values:
                                                    P_lm_blade_value = P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly)
                                                    P_lm_fuselage_value = P_lm_fuselage(random.choice(Cd_values), Af, Bh, V_lm_hfly)
                                                    P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly)
                                                    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                                                    T_l_hfly_value = T_l_hfly(random.choice(d_lm_hfly_values), V_lm_hfly)
                                                    P_l_hov_value = P_l_hov(Wl, P_l_b, Nr, Ar, Bh)
                                                    T_lm_hov_value = T_lm_hov(T_km_com_value, T_kml_up_value, T_ml_down_value)
                                                    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                                                    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                                                    results_matrix[i, j] = fitness_value

print("Fitness Results Matrix:")
print(results_matrix)
