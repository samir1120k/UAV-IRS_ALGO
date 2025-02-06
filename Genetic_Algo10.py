#update the previous algorithm
import pandas as pd
import numpy as np
import random
import math
from numpy import random

# Load datasets related to Base Stations, UAVs, and Clients
base = pd.read_csv(r'BS_data.csv')  # Dataset containing values related to Base Stations
uav = pd.read_csv(r'UAV_data.csv')  # Dataset containing values related to UAV-IRS
people = pd.read_csv(r'people_data.csv')  # Dataset containing values related to Clients

# Constants
Wl = [4, 5, 6, 7, 8, 9, 10, 6]  # Weight of UAV-IRS
H = base['H']  # Height of UAV-IRS above the ground
P_m_har = base['P_m_har']  # Harvesting power of each Base Station
T_m_har = base['T_m_har']  # Harvesting time of each Base Station
P_m_down = base['P_m_down']  # Downlink power of Base Station
T_ml_down = base['T_ml_down']  # Downlink time of Base Station to UAV-IRS
T_km_com = people['T_km_com']  # Computation time of each client
T_km_up = people['T_km_up']  # Uplink time of each client
V_lm_vfly = uav['V_lm_vfly']  # Velocity of UAV-IRS during vertical flight
V_lm_hfly = uav['V_lm_hfly']  # Velocity of UAV-IRS during horizontal flight
D_l_hfly = uav['D_l_hfly']  # Horizontal distance between UAV-IRS and Base Station

# Additional constants for calculations
delta = 2  # For calculating p_l_b
Ar = 0.503  # Area of rotor disc
s = 0.05  # Speed of rotor
Nr = 4  # Number of rotors for UAV-IRS
V_tip = 120  # Rotor solidity
Cd = 0.5  # Drag coefficient
Af = 0.06  # Fuselage area
D_km = 0.5  # Mbits

# Fitness function to calculate total energy consumption
def Fitness(E_ml_har, E_ml_down, E_ml_UAV):  # Equation number 30
    return E_ml_har + E_ml_down + E_ml_UAV  # Returns total energy consumption

# Energy consumption of the UAV-IRS
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):  # Equation number 20
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov  # Total energy consumption

# Power calculations for different flight modes
def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):  # Equation number 11
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):  # Equation number 13
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):  # Equation number 14
    return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):  # Equation number 15
    return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):  # Equation number 16
    return Wl * (np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2)**(1 / 2))

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):  # Equation number 18
    temp1 = Nr * P_l_b
    temp2 = abs(2 * (Nr * Bh * Ar))
    temp3 = np.sqrt(temp2)
    temp4 = (Wl**3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):  # Equation number 19
    return T_km_com + T_kml_up + T_ml_down  # Total time calculation

#_______________________________________________________________________________________________
# Genetic Algorithm Parameters
num_bs = 10
num_iterations = 10  # Number of iterations for each generation
num_generation = 10  # Number of generations
num_uav_irs = 8
all_best_combinations = []  # Store the best combination of BS and UAV

# Main Genetic Algorithm Loop
for l in range(num_bs):
    all_best_individuals = []
    P_m_har_value = P_m_har.values[l]
    T_m_har_value = T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    T_ml_down_value = T_ml_down.values[l]
    H_value = H.values[l]

    for k in range(num_uav_irs):
        best_fitness = float('inf')  # Initialize with infinity
        best_individual = {}  # Store the individual with the best fitness
        population = []  # Store the population of individuals
        V_lm_vfly_value = V_lm_vfly.values[k]
        V_lm_hfly_value = V_lm_hfly.values[k]
        D_l_hfly_value = D_l_hfly.values[k]
        Wl_value = Wl[k]

        for i in range(50):
            f_km_value = T_km_com.values[i]  # Randomly select computation time
            T_km_up_value = T_km_up.values[i]  # Randomly select uplink time

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
                        # Store parent data in the population list
            population.append({
                'fitness': result_fitness,
                'data': current_data
            })

        for j in range(num_generation):
            child_population=[]
            for i in range(0, len(population), 2): # Corrected loop to pair population members
                # Crossover Process
                parent1 = population[i]
                parent2 = population[i+1]

                child_data = {
                    key: parent1['data'][key] * 0.6 + parent2['data'][key] * (1 - 0.6)
                    for key in parent1['data']
                }

                # Mutation process
                u = np.random.uniform(0, 1, 1)[0]
                P_mutation = 0.5
                if u < P_mutation:
                    for key in child_data:
                        child_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0] # Reduced mutation scale for finer search

                # Compute fitness for child
                def compute_fitness(data):  # Takes a dictionary of parameters as input
                    H_value = data['H_value']
                    Wl_value = data['Wl_value']
                    P_m_down_value = data['P_m_down_value']
                    P_m_har_value = data['P_m_har_value']
                    T_m_har_value = data['T_m_har_value']
                    T_ml_down_value = data['T_ml_down_value']
                    f_km_value = data['f_km_value']
                    T_km_up_value = data['T_km_up_value']
                    V_lm_vfly_value = data['V_lm_vfly_value']
                    V_lm_hfly_value = data['V_lm_hfly_value']
                    D_l_hfly_value = data['D_l_hfly_value']

                    # Calculate Bh and p_l_b for child
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
                    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
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
                    return fitness_value, current_data

                child_fitness, child_data1 = compute_fitness(child_data)  # Compute child fitness

                child_population.append({
                    'fitness': child_fitness,
                    'data': child_data1
                })

            # create new population
            new_population = population + child_population #combine population and child population
            new_population = sorted(new_population, key=lambda x: x['fitness']) #sort based on fitness
            new_population = new_population[:50] #select top 50 based on fitness

            population = new_population #update population with new population


        best_individual_pair = population[0].copy() # Store a COPY
        best_individual_pair['generation'] = j + 1
        best_individual_pair['type'] = 'GA'
        all_best_individuals.append(best_individual_pair)  # Store a COPY

        all_best_combinations.append({
            'bs_index': l,
            'uav_index': k,
            'best_fitness': population[0]['fitness'], # Store best fitness for this BS-UAV pair
            'best_individual': best_individual_pair
        })
        # print(f"Best Fitness for BS {l}, UAV {k}: {population[0]['fitness']:.4f}") # Print fitness after each pair generation

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
print("\n--- Best Unique UAV Assignments (Auction Based Method) ---")
for assignment in best_assignments:
    print(f"\nBest Assignment for BS {assignment['bs_index']}:")
    print(f" UAV Index: {assignment['uav_index']}")
    best_ind = assignment['best_individual']
    print(f" Best Individual:")
    print(f" Generation: {best_ind['generation']}, Type: {best_ind['type']}")
    print(f" Fitness: {best_ind['fitness']:.4f}")
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