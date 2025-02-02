import pandas as pd
import numpy as np
import math
import random

df= pd.read_csv(r'BS_data.csv')
# Constants
Dm = 0.49  # Data in Mbits
D_km = 0.5  # Data in Mbits
B = 10  # Bandwidth in MHz
sigma_m = pow(10, -13)  # Noise variance (BS)
Tm = 10  # Time in seconds
sigma_km = pow(10, -13)  # Noise variance (Population)
j=2

# Define the number of base stations and UAV-IRS
num_base_stations = 10
num_uav_irs = 10
num_people = 50

# Initialize fitness matrix
fitness_matrix = np.zeros((num_base_stations, num_uav_irs))  # Shape: (10,10)

# Iterate over each base station and UAV-IRS pair
for m in range(num_base_stations):
    for l in range(num_uav_irs):

        # Generate random power and time values for energy harvesting
        P_m_har_values = df['P_m_har'] 
        T_m_har_values = df['T_m_har']

        # Energy harvested (returns scalar value after min selection)
        def E_ml_har(P_m_har, T_m_har):
            P_m_har=min(P_m_har)
            T_m_har=min(T_m_har)
            return P_m_har * T_m_har  # Returns a scalar after `min`

        E_ml_har_values = [E_ml_har(P_m_har_values,T_m_har_values)]
        E_ml_har_value = min(E_ml_har_values) # Returns a scalar
        minidx=E_ml_har_values.index(E_ml_har_value)
        print("Minimum Power: ",P_m_har_values[minidx])
        print("Minimum time: ",T_m_har_values[minidx])

#_____________________________________________________________________________________________
             # Transmission times (returns scalars after min selection)
        T_km_com_values = [D_km / random.uniform(20, 100) for _ in range(10)]
        T_kml_up_values = [Dm / random.uniform(1, 10) for _ in range(10)]
        T_ml_down_values = [Dm / random.uniform(1, 10) for _ in range(10)]
        P_ml_down_values = [Dm / random.uniform(1, 10) for _ in range(10)]

        T_km_com = min(T_km_com_values)  # Scalar
        T_kml_up = min(T_kml_up_values)  # Scalar
        T_ml_down = min(T_ml_down_values)  # Scalar
        P_m_down = min(P_ml_down_values)  # Scalar


        # Energy Consumption (Downlink) - Returns scalar after min selection
        def E_ml_down(P_m_down, T_ml_down):
            return P_m_down * T_ml_down  # Returns a scalar after `min`
        
        # T_km_com=D_km/F_km

        def P_m_down(Dm,T_km_com,T_kml_up,Tm,h_ml_worst): #eqation number 25
            temp1=pow(2,((Dm/T_km_com*T_kml_up*Tm)-1))
            return temp1/h_ml_worst


        E_ml_down_values = E_ml_down(P_m_down,T_ml_down)
        E_ml_down_value = min(E_ml_down_values)  # Returns a scalar
        # Energy consumption of UAV-IRS (Returns scalar after min selection)
        def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
            return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov  # Returns a scalar

        E_ml_UAV_values = [
            E_ml_UAV(random.uniform(0, 10), random.uniform(1, 10),
                     random.uniform(0, 10), random.uniform(1, 10),
                     random.uniform(0, 10), random.uniform(1, 10)) for _ in range(10)
        ]

        E_ml_UAV_value = min(E_ml_UAV_values)  # Returns a scalar

        # Compute final fitness value (returns scalar)
        def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
            return E_ml_har + E_ml_down + E_ml_UAV  # Returns a scalar

        fitness_matrix[m, l] = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

# Output the final fitness matrix
print(f"Final Fitness Matrix (Shape: {num_base_stations}x{num_uav_irs}):")
print(fitness_matrix)
