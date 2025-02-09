# coding: utf-8

#update the previous algorithm using PSO and GA and compare the result

import pandas as pd
import numpy as np
import random
import math
from numpy import random
import cmath
import matplotlib.pyplot as plt


# Load datasets related to Base Stations, UAVs, and Clients
base = pd.read_csv(r'BS_data.csv')
uav = pd.read_csv(r'UAV_data.csv')
people = pd.read_csv(r'people_data.csv')
IRS=pd.read_csv(r'IRS_data.csv')
IRS_UP=pd.read_csv(r'IRS_data_up.csv')

Angle_df=pd.read_csv(r'Angle.csv') # Renamed to avoid overwriting
h_l_km_df=pd.read_csv(r'h_l_km.csv') # Renamed to avoid overwriting
h_l_m_df=pd.read_csv(r'h_l_m.csv') # Renamed to avoid overwriting

Angle_UP_df=pd.read_csv(r'Angle.csv') # Renamed to avoid overwriting and using Angle_UP, corrected filename
g_l_km_df=pd.read_csv(r'h_l_km.csv') # Renamed to avoid overwriting and using g_l_km_df, corrected filename
g_l_m_df=pd.read_csv(r'h_l_m.csv') # Renamed to avoid overwriting and using g_l_m_df, corrected filename
f_km=pd.read_csv(r'f_km.csv')


# Constants
Wl = 35.28
H = 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
# T_ml_down = base['T_ml_down']
# T_km_com = people['T_km_com']
f_km=f_km['0']
# T_km_up = people['T_km_up']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly = 100
# Angle=IRS['Angle'] # Not needed as we load from CSV now
# h_l_km=IRS['h_l_km'] # Not needed as we load from CSV now
# h_l_m=IRS['h_l_m'] # Not needed as we load from CSV now
P_km_up=IRS_UP['P_km_up']
Angle1_col=IRS_UP['Angle'] # Renamed to avoid confusion with Angle dataframe
g_l_km_col=IRS_UP['h_l_km'] # Renamed
g_l_m_col=IRS_UP['h_l_m'] # Renamed


# Additional constants for calculations
delta = 2
Ar = 0.1256
s = 0.05
Nr = 4
V_tip = 102
Cd = 0.022
Af = 0.2113
D_km = 0.5
Dm=0.49
B=10 #MHz
sigma_km=10**(-13)
num_population=50


# Fitness function to calculate total energy consumption (common for GA and PSO)
def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    fitness_value = E_ml_har + E_ml_down + E_ml_UAV
    return max(0, fitness_value) # Ensure fitness is not negative

# Energy consumption of the UAV-IRS (common for GA and PSO)
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    e_uav_value = P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov
    return max(0, e_uav_value) # Ensure energy is not negative

# Power calculations for different flight modes (common for GA and PSO)
def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
    p_l_vfly_value = ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b
    return max(0, p_l_vfly_value) # Ensure power is not negative

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
    p_lm_hfly_value = P_lm_blade + P_lm_fuselage + P_lm_induced
    return max(0, p_lm_hfly_value) # Ensure power is not negative

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
    p_lm_blade_value = Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))
    return max(0, p_lm_blade_value) # Ensure power is not negative

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
    p_lm_fuselage_value = (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)
    return max(0, p_lm_fuselage_value) # Ensure power is not negative

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
    p_lm_induced_value = Wl * (np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2)**(1 / 2))
    return max(0, p_lm_induced_value) # Ensure power is not negative

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
    temp1 = Nr * P_l_b
    temp2 = abs(2 * (Nr * Bh * Ar))
    temp3 = np.sqrt(temp2)
    temp4 = (Wl**3 / 2) / temp3
    p_l_hov_value = temp1 + temp4
    return max(0, p_l_hov_value) # Ensure power is not negative

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
    t_lm_hov_value = T_km_com + T_kml_up + T_ml_down
    return max(0, t_lm_hov_value) # Ensure time is not negative

def R_ml_down(B,P_m_down,h_ml_worst): #equation number 7
    temp1=h_ml_worst*P_m_down
    rate_ml_down = B*math.log2(1+max(0, temp1)) # Ensure log argument is positive
    return max(0, rate_ml_down) # Ensure rate is not negative

def h_ml_worst(h_kml_down,sigma_km): #equation number 8
    h_ml_worst_value = h_kml_down/(sigma_km)
    return max(0, h_ml_worst_value) # Ensure channel gain is not negative

def calculate_exp_i_theta(theta): # part of equation 8
    return cmath.exp(1j * theta)
 # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8
    result=[]
    for i in range(len(Angle)):
        theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
    h_l_m_np = h_l_m.to_numpy() # Convert Series to numpy array
    h_l_km_np = h_l_km.to_numpy() # Convert Series to numpy array
    if h_l_m_np.ndim == 1:
        h_l_m_np = h_l_m_np.reshape(1, -1) # Reshape to 2D if necessary
    if h_l_km_np.ndim == 1:
        h_l_km_np = h_l_km_np.reshape(-1, 1) # Reshape to 2D if necessary


    a=np.dot(h_l_m_np,diagonal) # Use numpy arrays for dot product
    b=np.dot(a,h_l_km_np)       # Use numpy arrays for dot product
    final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
    h_kml_down_value = (final**2)
    return max(0, h_kml_down_value) # Ensure channel gain is not negative

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #equation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    log_arg = 1 + temp1
    if log_arg <= 0:
        log_arg = 1e-9  # Use a very small positive value if it's not positive
    rate_kml_up = B*math.log2(max(0, log_arg)) # Ensure log argument is positive
    return max(0, rate_kml_up) # Ensure rate is not negative

#this is inside the equation 4 have to take summation of h_i_up and P_i_up
def sub(P_i_up,h_il_up):
    sub_value = P_i_up*h_il_up
    return max(0, sub_value) # Ensure sub value is not negative


# Function to calculate energy and fitness (common for GA and PSO)
def calculate_energy_and_fitness(individual, l, k, Sub_value, algorithm_type):
    P_m_har_value = P_m_har.values[l]
    T_m_har_value = T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    H_value = H
    V_lm_vfly_value = V_lm_vfly.values[k]
    V_lm_hfly_value = V_lm_hfly.values[k]
    Wl_value = Wl

    f_km_value = individual['f_km_value']
    P_km_up_value = individual['P_km_up_value']

    # Calculate Bh and p_l_b
    Bh = (1 - 2.2558 * pow(10, 4) * H_value)
    Bh = max(1, Bh)
    p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

    # Calculate power values
    P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
    P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
    P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

    P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
    P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)


    # Calculate time values
    T_l_vfly_value = H_value / max(V_lm_vfly_value, 1e-9) # Avoid division by zero
    T_l_hfly_value = D_l_hfly / max(V_lm_hfly_value, 1e-9) # Avoid division by zero
    T_km_com_value = D_km / max(f_km_value, 1e-9) # Avoid division by zero

    Angle_row = Angle_df.iloc[0, :]  # Using same angle row for all, can be adjusted
    h_l_m_row = h_l_m_df.iloc[0, :]
    h_l_km_row = h_l_km_df.iloc[0, :]
    h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row)
    h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
    R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
    if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
        R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
    T_ml_down_value=Dm/max(R_ml_down_value, 1e-9) # Avoid division by zero

    Angle1_row = Angle_UP_df.iloc[0, :] # Using same angle row for UP link as well, can be adjusted
    g_l_m_row = g_l_m_df.iloc[0, :]
    g_l_km_row = g_l_km_df.iloc[0, :]
    h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Using same h_kml_down function for uplink

    R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
    T_km_up_value=Dm/max(R_kml_up_value, 1e-9) # equation number 5 # Avoid division by zero
    T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)


    # Calculate energy consumptions
    E_ml_har_value = P_m_har_value * T_m_har_value
    E_ml_down_value = P_m_down_value * T_ml_down_value
    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

    # Calculate fitness
    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

    energy_data = {
            'P_m_down_value': P_m_down_value,
            'P_m_har_value': P_m_har_value,
            'T_m_har_value': T_m_har_value,
            'T_ml_down_value': T_ml_down_value,
            'f_km_value': f_km_value,
            'T_km_up_value': T_km_up_value,
            'V_lm_vfly_value': V_lm_vfly_value,
            'V_lm_hfly_value': V_lm_hfly_value,
            'P_km_up_value': P_km_up_value,
            'h_kml_down_value': h_kml_down_value,
            'T_km_com_value': T_km_com_value
            }

    return fitness_value, energy_data



def run_ga(num_bs, num_uav_irs, num_generation, population_size, f_km, P_km_up, V_lm_vfly, V_lm_hfly):
    all_best_combinations_ga = []
    all_best_individuals_ga = []

    for l in range(num_bs):
        all_best_individuals_bs_ga = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness_ga = float('inf')
            best_individual_ga = {}
            population_ga = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            for i in range(num_population):
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)
                Sub_value = max(0, Sub_value) # Ensure Sub_value is not negative


            # Initialize population
            for i in range(population_size):
                f_km_value = f_km[i]
                P_km_up_value = P_km_up.values[i]

                initial_individual = {
                        'f_km_value': f_km_value,
                        'P_km_up_value': P_km_up_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value
                        }

                fitness_value_ga, energy_data_ga = calculate_energy_and_fitness(initial_individual, l, k, Sub_value, 'GA')


                # Store initial population data
                population_ga.append({
                        'fitness': fitness_value_ga,
                        'data':  energy_data_ga
                        })

            generations_data_ga = []
            for j in range(num_generation):
                child_population_ga = []
                for i in range(0, population_size, 2): # Loop through population with step of 2
                    # Crossover
                    parent1_ga = population_ga[i]
                    parent2_ga = population_ga[i+1]
                    child_data_ga = {}
                    for key in parent1_ga['data']:
                        child_data_ga[key] = parent1_ga['data'][key] * 0.6 + parent2_ga['data'][key] * (1 - 0.6)

                    # Mutation
                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5  # Increased mutation rate to 0.5 for better exploration
                    if u < P_mutation:
                        for key in child_data_ga:
                            child_data_ga[key] += random.normal(loc=0, scale=1, size=(1))[0]
                            # child_data_ga[key] = max(0, child_data_ga[key]) # Ensure value is not negative after mutation

                    # Compute child fitness
                    fitness_value_child_ga, child_energy_data_ga = calculate_energy_and_fitness(child_data_ga, l, k, Sub_value, 'GA')
                    child_population_ga.append({'fitness': fitness_value_child_ga, 'data': child_energy_data_ga})

                # Create new population
                new_population_ga = population_ga + child_population_ga
                new_population_ga = sorted(new_population_ga, key=lambda x: x['fitness'])
                population_ga = new_population_ga[:population_size]
                generations_data_ga.append(population_ga[0].copy())


            best_individual_pair_ga = population_ga[0].copy()
            best_individual_pair_ga['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair_ga['type'] = 'GA'
            best_individual_pair_ga['bs_index'] = l
            best_individual_pair_ga['uav_index'] = k
            all_best_individuals_bs_ga.append(best_individual_pair_ga)

            all_best_combinations_ga.append({
                    'bs_index': l,
                    'uav_index': k,
                    'best_fitness': population_ga[0]['fitness'],
                    'best_individual': best_individual_pair_ga,
                    'generation_fitness': [gen['fitness'] for gen in generations_data_ga]
                    })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs_ga = min(all_best_individuals_bs_ga, key=lambda x: x['fitness'])


    return all_best_combinations_ga


def run_pso(num_bs, num_uav_irs, num_generation, population_size, f_km, P_km_up, V_lm_vfly, V_lm_hfly):
    all_best_combinations_pso = []
    all_best_individuals_pso = []

    # PSO parameters
    inertia_weight = 0.5
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    for l in range(num_bs):
        all_best_individuals_bs_pso = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness_pso = float('inf')
            best_individual_pso = {}
            swarm = []
            global_best_fitness_pso = float('inf')
            global_best_position_pso = None
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            for i in range(num_population):
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)
                Sub_value = max(0, Sub_value) # Ensure Sub_value is not negative


            # Initialize swarm
            for _ in range(population_size):
                f_km_value = random.choice(f_km)
                P_km_up_value = random.choice(P_km_up.values)

                # Initialize position and velocity for each particle
                position = {
                        'f_km_value': f_km_value,
                        'P_km_up_value': P_km_up_value,
                        'V_lm_vfly_value': V_lm_vfly_value, # Fixed for each UAV index
                        'V_lm_hfly_value': V_lm_hfly_value  # Fixed for each UAV index
                        }
                velocity = {
                        'f_km_value': 0.0,
                        'P_km_up_value': 0.0
                        }
                personal_best_fitness_pso = float('inf')
                personal_best_position_pso = position.copy()

                swarm.append({
                        'position': position,
                        'velocity': velocity,
                        'personal_best_fitness': personal_best_fitness_pso,
                        'personal_best_position': personal_best_position_pso
                        })

            generations_data_pso = []
            for j in range(num_generation):
                for particle in swarm:
                    # Get current position values
                    f_km_value = particle['position']['f_km_value']
                    P_km_up_value = particle['position']['P_km_up_value']
                    V_lm_vfly_value = particle['position']['V_lm_vfly_value']
                    V_lm_hfly_value = particle['position']['V_lm_hfly_value']

                    current_individual = {
                            'f_km_value': f_km_value,
                            'P_km_up_value': P_km_up_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value
                            }

                    # Calculate fitness
                    current_fitness_pso, energy_data_pso = calculate_energy_and_fitness(current_individual, l, k, Sub_value, 'PSO')

                    # Update personal best
                    if current_fitness_pso < particle['personal_best_fitness']:
                        particle['personal_best_fitness'] = current_fitness_pso
                        particle['personal_best_position'] = particle['position'].copy()

                    # Update global best
                    if current_fitness_pso < global_best_fitness_pso:
                        global_best_fitness_pso = current_fitness_pso
                        global_best_position_pso = particle['position'].copy()


                generations_data_pso.append({'fitness': global_best_fitness_pso, 'data': global_best_position_pso.copy()})


                # Update particle velocities and positions
                for particle in swarm:
                    # Update velocities
                    particle['velocity']['f_km_value'] = (
                            inertia_weight * particle['velocity']['f_km_value'] +
                            cognitive_coefficient * random.random() * (particle['personal_best_position']['f_km_value'] - particle['position']['f_km_value']) +
                            social_coefficient * random.random() * (global_best_position_pso['f_km_value'] - particle['position']['f_km_value'])
                    )
                    particle['velocity']['P_km_up_value'] = (
                            inertia_weight * particle['velocity']['P_km_up_value'] +
                            cognitive_coefficient * random.random() * (particle['personal_best_position']['P_km_up_value'] - particle['position']['P_km_up_value']) +
                            social_coefficient * random.random() * (global_best_position_pso['P_km_up_value'] - particle['position']['P_km_up_value'])
                    )

                    # Update positions
                    particle['position']['f_km_value'] += particle['velocity']['f_km_value']
                    particle['position']['P_km_up_value'] += particle['velocity']['P_km_up_value']
                    particle['position']['f_km_value'] = max(0, particle['position']['f_km_value']) # Ensure value is not negative
                    particle['position']['P_km_up_value'] = max(0, particle['position']['P_km_up_value']) # Ensure value is not negative


            best_individual_pair_pso = generations_data_pso[-1].copy() # Take the best from the last generation
            best_individual_pair_pso['generation'] = j + 1 #Actually it is last generation number, but for consistency let's keep j+1
            best_individual_pair_pso['type'] = 'PSO'
            best_individual_pair_pso['bs_index'] = l
            best_individual_pair_pso['uav_index'] = k
            all_best_individuals_bs_pso.append(best_individual_pair_pso)


            all_best_combinations_pso.append({
                    'bs_index': l,
                    'uav_index': k,
                    'best_fitness': global_best_fitness_pso,
                    'best_individual': best_individual_pair_pso,
                    'generation_fitness': [gen['fitness'] for gen in generations_data_pso]
                    })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs_pso = min(all_best_individuals_bs_pso, key=lambda x: x['fitness'])

    return all_best_combinations_pso


def select_best_assignments(all_best_combinations, num_bs, num_uav_irs):
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])
    return best_assignments


# Genetic Algorithm Parameters
num_bs = 5
num_generation_ga = 30  # Number of generations for GA
num_uav_irs = 8
population_size_ga = 50  # Population size for GA


# Particle Swarm Optimization Parameters
num_generation_pso = 30  # Number of iterations for PSO
population_size_pso = 50  # Swarm size for PSO


# Run Genetic Algorithm
all_best_combinations_ga = run_ga(num_bs, num_uav_irs, num_generation_ga, population_size_ga, f_km, P_km_up, V_lm_vfly, V_lm_hfly)
best_assignments_ga = select_best_assignments(all_best_combinations_ga, num_bs, num_uav_irs)


# Run Particle Swarm Optimization
all_best_combinations_pso = run_pso(num_bs, num_uav_irs, num_generation_pso, population_size_pso, f_km, P_km_up, V_lm_vfly, V_lm_hfly)
best_assignments_pso = select_best_assignments(all_best_combinations_pso, num_bs, num_uav_irs)


# --- Print Best Assignments and Plotting ---
algorithms = {'GA': {'best_assignments': best_assignments_ga, 'num_generation': num_generation_ga, 'all_best_combinations': all_best_combinations_ga},
                'PSO': {'best_assignments': best_assignments_pso, 'num_generation': num_generation_pso, 'all_best_combinations': all_best_combinations_pso}}

best_pair_for_plot_comparison = {}
min_fitness_for_plot_comparison = {'GA': float('inf'), 'PSO': float('inf')}


for algorithm_name, algorithm_data in algorithms.items():
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method with {algorithm_name}) ---")
    best_assignments = algorithm_data['best_assignments']
    num_generation = algorithm_data['num_generation']
    all_best_combinations = algorithm_data['all_best_combinations']
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f"  UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f"  Best Individual:")
        print(f"   Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}")
        for key, value in best_ind['data'].items():
            print(f"    {key}: {value:.4f}")
        print("-" * 20)

        if assignment['best_individual']['fitness'] < min_fitness_for_plot:
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    if best_pair_for_plot:
        best_pair_for_plot_comparison[algorithm_name] = best_pair_for_plot
        min_fitness_for_plot_comparison[algorithm_name] = min_fitness_for_plot
    else:
        print(f"\nNo best pair found for plotting for {algorithm_name}.")


# --- Plotting Comparison Graphs ---
plt.figure(figsize=(12, 7))

# Plot Fitness for Best Pair for each algorithm
for algorithm_name, best_pair_for_plot in best_pair_for_plot_comparison.items():
    fitness_history_best_pair = best_pair_for_plot['generation_fitness']
    num_generation = algorithms[algorithm_name]['num_generation']
    generations = range(1, num_generation + 1)
    plt.plot(generations, fitness_history_best_pair, marker='o', linestyle='-', label=f'Best Pair {algorithm_name} (BS {best_pair_for_plot['bs_index']}, UAV {best_pair_for_plot['uav_index']})')


plt.xlabel('Generation/Iteration')
plt.ylabel('Fitness Value')
plt.title('Fitness Improvement Over Generations/Iterations for Best BS-UAV Pair (GA vs PSO)')
plt.grid(True)
plt.legend()
plt.xticks(range(0, max(num_generation_ga, num_generation_pso) + 2, 5))
plt.tight_layout()
plt.show()



# Calculate and Print Percentage Improvement
if min_fitness_for_plot_comparison['GA'] != float('inf') and min_fitness_for_plot_comparison['PSO'] != float('inf'):
    if min_fitness_for_plot_comparison['GA'] < min_fitness_for_plot_comparison['PSO']:
        better_algorithm = 'GA'
        worse_algorithm = 'PSO'
        improvement = ((min_fitness_for_plot_comparison['PSO'] - min_fitness_for_plot_comparison['GA']) / min_fitness_for_plot_comparison['PSO']) * 100
    else:
        better_algorithm = 'PSO'
        worse_algorithm = 'GA'
        improvement = ((min_fitness_for_plot_comparison['GA'] - min_fitness_for_plot_comparison['PSO']) / min_fitness_for_plot_comparison['GA']) * 100

    print(f"\n--- Performance Comparison ---")
    print(f"Best Fitness with GA: {min_fitness_for_plot_comparison['GA']:.4f}")
    print(f"Best Fitness with PSO: {min_fitness_for_plot_comparison['PSO']:.4f}")
    print(f"\n{better_algorithm} performed better than {worse_algorithm} by {improvement:.2f}% in terms of fitness value.")
else:
    print("\nCould not compare performance as best fitness was not found for both algorithms.")


# Sum of Best Fitness Values Across Generations for Comparison
plt.figure(figsize=(12, 7))

for algorithm_name, algorithm_data in algorithms.items():
    sum_fitness_per_generation = [0] * algorithms[algorithm_name]['num_generation']
    all_best_combinations = algorithms[algorithm_name]['all_best_combinations']
    num_generation = algorithms[algorithm_name]['num_generation']

    for gen_idx in range(num_generation):
        generation_sum = 0
        for combination in all_best_combinations:
            generation_sum += combination['generation_fitness'][gen_idx]
        sum_fitness_per_generation[gen_idx] = generation_sum

    generation_indices = list(range(1, num_generation + 1))
    plt.plot(generation_indices, sum_fitness_per_generation, marker='o', linestyle='-', label=f'Sum Fitness {algorithm_name}')


plt.title('Sum of Best Fitness Values Across Generations/Iterations (GA vs PSO)')
plt.xlabel('Generation/Iteration Number')
plt.ylabel('Sum of Best Fitness Values')
plt.grid(True)
plt.legend()
plt.xticks(range(0, max(num_generation_ga, num_generation_pso) + 2, 5))
plt.show()