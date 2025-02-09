import pandas as pd
import numpy as np
import random

# Load datasets (replace with your actual paths)
base = pd.read_csv(r'BS_data.csv')
uav = pd.read_csv(r'UAV_data.csv')
people = pd.read_csv(r'people_data.csv')

# Constants
Wl = 35.28
H = 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
T_ml_down = base['T_ml_down']
T_km_com = people['T_km_com']
T_km_up = people['T_km_up']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly = uav['D_l_hfly']
delta = 2
Ar = 0.503
s = 0.05
Nr = 4
V_tip = 120
Cd = 0.5
Af = 0.06
D_km = 0.5

# Fitness function
def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    return E_ml_har + E_ml_down + E_ml_UAV

# Energy consumption of the UAV-IRS
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# Power calculations
def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
    return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
    return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
    return Wl * (np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2)**(1 / 2))

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
    temp1 = Nr * P_l_b
    temp2 = abs(2 * (Nr * Bh * Ar))
    temp3 = np.sqrt(temp2)
    temp4 = (Wl**3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
    return T_km_com + T_kml_up + T_ml_down

# Genetic Algorithm Parameters
num_bs = 10
num_iterations = 10  # Number of iterations for each generation
num_generation = 10  # Number of generations
num_uav_irs = 8
all_best_combinations=[]
all_best_individuals = []


# Main Genetic Algorithm Loop
for l in range(num_bs):
    all_best_individuals = []
    P_m_har_value = P_m_har.values[l]
    T_m_har_value = T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    T_ml_down_value = T_ml_down.values[l]
    H_value = H.values[l]

    overall_best_fitness = float('inf')  # Initialize overall best fitness
    overall_best_individual = {}  # Store the overall best individual

    for k in range(num_uav_irs):
        best_fitness = float('inf')  # Initialize with infinity
        best_individual = {}  # Store the individual with the best fitness
        V_lm_vfly_value = V_lm_vfly.values[k]
        V_lm_hfly_value = V_lm_hfly.values[k]
        D_l_hfly_value = D_l_hfly.values[k]
        Wl_value = Wl[k]

        for i in range(num_iterations):
            f_km_value = random.choice(T_km_com.values)  # Randomly select computation time
            T_km_up_value = random.choice(T_km_up.values)  # Randomly select uplink time

            # Calculate Bh and p_l_b
            Bh = (1 - 2.2558 * pow(10, 4) * H_value)
            Bh = max(1, Bh)
            p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value * V_lm_hfly_value
            E_ml_har_value = P_m_har_value * T_m_har_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness
            result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            # Store current data
            current_data = {  # Store all relevant variables for this iteration
                'H_value': H_value,
                'Wl_value': Wl_value,
                'P_m_down_value': P_m_down_value,
                'P_m_har_value': P_m_har_value,
                'T_m_har_value': T_m_har_value,
                'T_ml_down_value': T_ml_down_value,
                'f_km_value': f_km_value,
                'T_km_up_value': T_km_up_value,
                'V_lm_vfly_value': V_lm_vfly_value,
                'V_lm_hfly_value': V_lm_hfly_value,
                'D_l_hfly_value': D_l_hfly_value,
            }


            if result_fitness < best_fitness:
                best_fitness = result_fitness
                best_individual = current_data.copy()

        # Check if the current UAV's best is the overall best
        if best_fitness < overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_individual = best_individual.copy()
        all_best_combinations.append({  # Store the best for this BS-UAV pair
            'bs_index': l,
            'uav_index': k,
            'best_fitness': best_fitness,
            'best_individual': best_individual.copy()
    })

# Print the results for all BS-UAV pairs
# print("\n--- Best Combinations for all BS-UAV Pairs ---")
# for combination in all_best_combinations:
#     print(f"\nBase Station: {combination['bs_index'] + 1}, UAV: {combination['uav_index'] + 1}")  # +1 for user-friendly indexing
#     print(f"Best Fitness: {combination['best_fitness']}")
#     for key, value in combination['best_individual'].items():
#         print(f"  {key}: {value}")
#     print("-" * 20)


#select the best unique Base station and UAV-IRS pair using Auction based method
# Optimization: Create a dictionary to quickly lookup combinations by BS index and UAV index
combination_lookup = {}
for combination in all_best_combinations:
    if combination['bs_index'] not in combination_lookup:
        combination_lookup[combination['bs_index']] = {}
    combination_lookup[combination['bs_index']][combination['uav_index']] = combination

# Assign best UAV to each BS (ensuring unique assignments)
best_assignments = []
unassigned_bs = list(range(num_bs))  # List of unassigned BS indices
unassigned_uavs = list(range(num_uav_irs))  # List of unassigned UAV indices

while unassigned_bs and unassigned_uavs:
    best_combination_overall = None

    for l in unassigned_bs:
        best_fitness_for_bs = float('inf')
        best_combination_for_bs = None
        for k in unassigned_uavs:
            # Use the lookup dictionary for O(1) access instead of iterating through all combinations
            if l in combination_lookup and k in combination_lookup[l]:  # Check if BS and UAV are in the lookup
                combination = combination_lookup[l][k]
                if combination['best_individual']['fitness'] < best_fitness_for_bs:
                    best_fitness_for_bs = combination['best_individual']['fitness']
                    best_combination_for_bs = combination

        if best_combination_for_bs:
            if best_combination_overall is None or best_combination_for_bs['best_individual']['fitness'] < best_assignments[-1]['best_individual']['fitness'] if best_assignments else best_combination_for_bs['best_individual']['fitness'] < float('inf'):
                best_combination_overall = best_combination_for_bs

    if best_combination_overall:
        best_assignments.append(best_combination_overall)
        unassigned_bs.remove(best_combination_overall['bs_index'])
        unassigned_uavs.remove(best_combination_overall['uav_index'])

# Print the best assignments
print("\n--- Best UAV Assignments (Ensuring Unique Assignments) ---")
for assignment in best_assignments:
    print(f"\nBest Assignment for BS {assignment['bs_index']}:")
    print(f" UAV Index: {assignment['uav_index']}")
    best_ind = assignment['best_individual']
    print(f" Best Individual:")
    print(f" Generation: {best_ind['generation']}, Type: {best_ind['type']}")
    print(f" Fitness: {best_ind['fitness']}")
    for key, value in best_ind['data'].items():
        print(f" {key}: {value}")
    print("-" * 20)

# Print base stations or UAVs without assignments (if any)
if unassigned_bs:
    print("\n--- Base Stations without Assigned UAVs ---")
    for bs_index in unassigned_bs:
        print(f"BS {bs_index} : No UAV is assigned")
        print("-" * 20)

if unassigned_uavs:
    print("\n--- UAVs without Assigned Base Stations ---")
    for uav_index in unassigned_uavs:
        print(f"UAV {uav_index} : No BS is assigned")
        print("-" * 20)