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
V_l_vfly_values = np.linspace(1, 100, 10)  # vertical flight speed values
V_lm_hfly_values = np.linspace(1, 100, 10)  # horizontal flight speed values
H_values = np.linspace(1, 100, 10)  # height values
d_lm_hfly_values = np.linspace(0, 100, 10)  # horizontal distance values
F_km_values = np.linspace(20, 100, 10)  # computation capacities

# Functions
def E_ml_har(P_m_har, T_m_har):
    """Calculates energy harvesting matrix"""
    return P_m_har * T_m_har


def R_kml_up(B, P_km_up, h_kml_up, P_i_up, h_il_up, sigma_m):
    """Calculates uplink data rate matrix"""
    temp1 = (P_km_up * h_kml_up) / ((P_i_up * h_il_up) + pow(sigma_m, 2))
    return B * np.log2(1 + temp1)

def T_km_com(D_km, F_km):
    """Calculates computation time for users"""
    return D_km / F_km

def T_kml_up(Dm, R_kml_up):
    """Calculates uplink transmission time matrix"""
    return Dm / R_kml_up

def h_kml_up(g_lm, V_lm_up, g_km_l):
    """Calculates uplink channel gain matrix (Corrected)"""
    # g_lm: l x 1, V_lm_up: scalar, g_km_l: 1 x l
    return abs(g_lm @ np.array([[V_lm_up]]) @ g_km_l) # Use @ for matrix multiplication

def h_kml_down(h_lm, V_lm_down, h_km_l):
    """Calculates downlink channel gain matrix (Corrected)"""
    # h_lm: l x 1, V_lm_down: scalar, h_km_l: 1 x l
    return abs(h_lm @ np.array([[V_lm_down]]) @ h_km_l) # Use @ for matrix multiplication


def P_m_down(Dm, T_km_com, T_kml_up, Tm, h_ml_worst):
    """Calculates downlink transmission power (Logarithm method)"""
    try:
        exponent = (Dm / (T_km_com * T_kml_up * Tm)) - 1
        temp1 = np.exp(exponent * np.log(2))
        return temp1 / h_ml_worst
    except ZeroDivisionError:
        print("Warning: Division by zero in P_m_down. Check T_km_com, T_kml_up, Tm.")
        return 0  # Or handle it differently
    except OverflowError:
        print("Warning: Overflow in P_m_down. Check input values.")
        return 0 # Or handle it differently

def E_ml_down(P_m_down, T_ml_down):
    """Calculates downlink energy consumption (NaN check)"""
    result = P_m_down * T_ml_down
    if np.isnan(result) or np.isinf(result):  # Check for both NaN and infinity
        print("Warning: E_ml_down is NaN or inf. Check P_m_down and T_ml_down.")
        return 0  # Or handle it differently
    return result

def h_ml_worst(h_kml_down, sigma_km):
    """Calculates worst-case downlink channel gain"""
    return h_kml_down / pow(sigma_km, 2)  # Corrected: removed extra division by sigma_km



def R_ml_down(B, P_m_down, h_ml_worst):
    """Calculates downlink data rate matrix"""
    temp1 = h_ml_worst * P_m_down
    return B * np.log2(1 + temp1)

def T_ml_down(Dm, R_ml_down):
    """Calculates downlink transmission time matrix"""
    return Dm / R_ml_down

def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
    """Calculates power consumption during vertical flight"""
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(pow(V_l_vfly, 2) + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def T_l_vfly(H, V_l_vfly):
    """Calculates time for vertical flight"""
    return H / V_l_vfly

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
    """Calculates power consumption by blades during horizontal flight"""
    return Nr * P_l_b * (1 + ((3 * pow(V_lm_hfly, 2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
    """Calculates power consumption by fuselage during horizontal flight"""
    return (1 / 2) * Cd * Af * Bh * pow(V_lm_hfly, 3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
    """Calculates induced power consumption during horizontal flight"""
    return Wl * np.sqrt(pow(Wl, 2) / (4 * pow(Nr, 2) * pow(Bh, 2) * pow(Ar, 2)) + (pow(V_lm_hfly, 4) / 4) - (pow(V_lm_hfly, 2) / 2))

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
    """Calculates total power consumption during horizontal flight"""
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def T_l_hfly(d_lm_hfly, V_lm_hfly):
    """Calculates time for horizontal flight"""
    return d_lm_hfly / V_lm_hfly

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
    """Calculates power consumption during hovering"""
    temp1 = Nr * P_l_b
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(2 * temp2)
    temp4 = pow(Wl, 3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
    """Calculates hovering time"""
    return T_km_com + T_kml_up + T_ml_down

def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    """Calculates total energy consumption of the UAV-IRS"""
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    """Calculates total fitness value (energy consumption)"""
    return E_ml_har + E_ml_down + E_ml_UAV

# Initialize parameters (reduced ranges for faster execution)
P_m_har_values = np.linspace(1, 100, 5)
T_m_har_values = np.linspace(1, 100, 5)
P_km_up_values = np.linspace(1, 100, 5)
P_i_up_values = np.linspace(1, 100, 5)
g_ml_values = np.linspace(1, 100, 5)
g_km_l_values = np.linspace(1, 100, 5)
h_il_up_values = np.linspace(1, 100, 5)
h_lm_values = np.linspace(1, 100, 5)
h_km_l_values = np.linspace(1, 100, 5)


# Initialize matrices
results_matrix = np.zeros((len(P_m_har_values), len(T_m_har_values)))  # Corrected shape

# Main loop to calculate fitness
for i, P_m_har in enumerate(P_m_har_values):
    for j, T_m_har in enumerate(T_m_har_values):
        fitness_values = []
        for Wl in Wl_values:
            for V_l_vfly in V_l_vfly_values:
                for V_lm_hfly in V_lm_hfly_values:
                    Bh = (1 - 2.2558 * pow(10, -4) * random.choice(H_values))
                    P_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)
                    P_l_vfly_value = P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh)
                    T_l_vfly_value = T_l_vfly(random.choice(H_values), V_l_vfly)
                    P_lm_blade_value = P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly)
                    P_lm_fuselage_value = P_lm_fuselage(random.choice(Cd_values), Af, Bh, V_lm_hfly)
                    P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly)
                    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                    T_l_hfly_value = T_l_hfly(random.choice(d_lm_hfly_values), V_lm_hfly)
                    P_l_hov_value = P_l_hov(Wl, P_l_b, Nr, Ar, Bh)

                    # Inner loops for channel parameters
                    min_fitness = float('inf') # Initialize with infinity to find minimum
                    for k, P_km_up in enumerate(P_km_up_values):
                        for m, P_i_up in enumerate(P_i_up_values):
                            for n, g_ml_scalar in enumerate(g_ml_values):
                                for o, g_km_l_scalar in enumerate(g_km_l_values):
                                    for p, h_il_up in enumerate(h_il_up_values):
                                        for q, h_lm_scalar in enumerate(h_lm_values):
                                            for r, h_km_l_scalar in enumerate(h_km_l_values):
                                                for l_index, F_km in enumerate(F_km_values):  # F_km loop INSIDE

                                                    l_dim = 5  # Example dimension for 'l' (number of users).  CRUCIAL!

                                                    g_lm = np.full((l_dim, 1), g_ml_scalar)  # l x 1
                                                    g_km_l = np.full((1, l_dim), g_km_l_scalar)  # 1 x l
                                                    h_lm = np.full((l_dim, 1), h_lm_scalar)  # l x 1
                                                    h_km_l = np.full((1, l_dim), h_km_l_scalar)  # 1 x l

                                                    V_lm_up = np.exp(np.radians(random.uniform(0, 180)))  # Scalar
                                                    V_lm_down = np.exp(np.radians(random.uniform(0, 180)))  # Scalar

                                                    h_kml_up_matrix = h_kml_up(g_lm, V_lm_up, g_km_l)  # l x l matrix
                                                    h_kml_down_matrix = h_kml_down(h_lm, V_lm_down, h_km_l)  # l x l matrix

                                                    h_kml_up_value = np.mean(h_kml_up_matrix)  # Scalar
                                                    h_kml_down_value = np.mean(h_kml_down_matrix)  # Scalar

                                                    R_kml_up_value = R_kml_up(B, P_km_up, h_kml_up_value, P_i_up, h_il_up, sigma_m)  # Scalar

                                                    T_km_com_value = T_km_com(D_km, F_km)  # Scalar
                                                    T_kml_up_value = T_kml_up(Dm, R_kml_up_value)  # Scalar

                                                    h_ml_worst_value = h_ml_worst(h_kml_down_value, sigma_km)  # Scalar
                                                    P_m_down_value = P_m_down(Dm, T_km_com_value, T_kml_up_value, Tm, h_ml_worst_value)  # Scalar
                                                    R_ml_down_value = R_ml_down(B, P_m_down_value, h_ml_worst_value)  # Scalar
                                                    T_ml_down_value = T_ml_down(Dm, R_ml_down_value)  # Scalar

                                                    E_ml_har_value = E_ml_har(P_m_har, T_m_har)  # Scalar
                                                    E_ml_down_value = E_ml_down(P_m_down_value, T_ml_down_value)  # Scalar

                                                    T_lm_hov_value = T_lm_hov(T_km_com_value, T_kml_up_value, T_ml_down_value)  # Scalar
                                                    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)  # Scalar
                                                    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)  # Scalar
                                                    min_fitness = min(min_fitness, fitness_value)

        results_matrix[i, j] = min_fitness

print("Fitness Results Matrix:")
print(results_matrix)