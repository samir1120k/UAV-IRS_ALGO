# coding: utf-8
#update the previous algorithm using PSO
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

# Constants
Wl = 35.28
H = 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
# T_ml_down = base['T_ml_down']
# T_km_com = people['T_km_com']
f_km=np.random.uniform(0,10, 50) #Frequency of local computing
# T_km_up = people['T_km_up']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly = 100
Angle=IRS['Angle']
h_l_km=IRS['h_l_km']
h_l_m=IRS['h_l_m']
P_km_up=IRS_UP['P_km_up']
Angle1=IRS_UP['Angle']
g_l_km=IRS_UP['h_l_km']
g_l_m=IRS_UP['h_l_m']


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

# Fitness function to calculate total energy consumption
def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    return E_ml_har + E_ml_down + E_ml_UAV

# Energy consumption of the UAV-IRS
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# Power calculations for different flight modes
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

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=h_ml_worst*P_m_down
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
                                    # the value for each itaration

def calculate_exp_i_theta(theta): # part of equation 8
    return cmath.exp(1j * theta)
 # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8

    result=[]
    for i in range(len(Angle)):
        theta_radians = math.radians(Angle[i])
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
    # h_l_m_np = np.array(eval(h_l_m.values[0]))  # Convert string to numpy array
    # h_l_km_np = np.array(eval(h_l_km.values[0])) # Convert string to numpy array

    a=np.dot(h_l_m,diagonal) # Use numpy arrays for dot product
    b=np.dot(a,h_l_km)       # Use numpy arrays for dot product
    final=abs(b)
    return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    log_arg = 1 + temp1
    if log_arg <= 0:
        log_arg = 1e-9  # Use a very small positive value if it's not positive
    return B*math.log2(log_arg)
    


#_______________________________________________________________________________________________
# Particle Swarm Optimization Parameters
num_bs = 5
num_generation = 30  # Number of iterations for PSO
num_uav_irs = 8
population_size = 50  # Swarm size for PSO
all_best_combinations = []
all_best_individuals = []

# PSO parameters
inertia_weight = 0.5
cognitive_coefficient = 1.5
social_coefficient = 1.5

# Main Particle Swarm Optimization Loop
for l in range(num_bs):
    all_best_individuals_bs = []
    P_m_har_value = P_m_har.values[l]
    T_m_har_value = T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    H_value = H

    for k in range(num_uav_irs):
        best_fitness = float('inf')
        best_individual = {}
        swarm = []
        global_best_fitness = float('inf')
        global_best_position = None
        V_lm_vfly_value = V_lm_vfly.values[k]
        V_lm_hfly_value = V_lm_hfly.values[k]
        D_l_hfly_value = D_l_hfly
        Wl_value = Wl

        # Initialize swarm
        for _ in range(population_size):
            f_km_value = random.choice(f_km)
            P_km_up_value = random.choice(P_km_up.values)

            # Initialize position and velocity for each particle
            position = {
                'f_km_value': f_km_value,
                'P_km_up_value': P_km_up_value,
                'V_lm_vfly_value': V_lm_vfly_value, # Fixed for each UAV index
                'V_lm_hfly_value': V_lm_hfly_value # Fixed for each UAV index
            }
            velocity = {
                'f_km_value': 0.0,
                'P_km_up_value': 0.0
            }
            personal_best_fitness = float('inf')
            personal_best_position = position.copy()

            swarm.append({
                'position': position,
                'velocity': velocity,
                'personal_best_fitness': personal_best_fitness,
                'personal_best_position': personal_best_position
            })

        generations_data = []
        for j in range(num_generation):
            for particle in swarm:
                # Calculate Bh and p_l_b
                Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                Bh = max(1, Bh)
                p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                # Get current position values
                f_km_value = particle['position']['f_km_value']
                P_km_up_value = particle['position']['P_km_up_value']
                V_lm_vfly_value = particle['position']['V_lm_vfly_value']
                V_lm_hfly_value = particle['position']['V_lm_hfly_value']


                # Calculate power values
                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

                # Calculate time and energy values
                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value * V_lm_hfly_value
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(Angle,h_l_m,h_l_km)
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1,g_l_m,g_l_km)
                Sub_value=20 #this is inside the equation 4 have to take summation of h_i_up and P_i_up
                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                current_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Update personal best
                if current_fitness < particle['personal_best_fitness']:
                    particle['personal_best_fitness'] = current_fitness
                    particle['personal_best_position'] = particle['position'].copy()

                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particle['position'].copy()

            generations_data.append({'fitness': global_best_fitness, 'data': global_best_position.copy()})

            # Update particle velocities and positions
            for particle in swarm:
                # Update velocities
                particle['velocity']['f_km_value'] = (
                    inertia_weight * particle['velocity']['f_km_value'] +
                    cognitive_coefficient * random.random() * (particle['personal_best_position']['f_km_value'] - particle['position']['f_km_value']) +
                    social_coefficient * random.random() * (global_best_position['f_km_value'] - particle['position']['f_km_value'])
                )
                particle['velocity']['P_km_up_value'] = (
                    inertia_weight * particle['velocity']['P_km_up_value'] +
                    cognitive_coefficient * random.random() * (particle['personal_best_position']['P_km_up_value'] - particle['position']['P_km_up_value']) +
                    social_coefficient * random.random() * (global_best_position['P_km_up_value'] - particle['position']['P_km_up_value'])
                )

                # Update positions
                particle['position']['f_km_value'] += particle['velocity']['f_km_value']
                particle['position']['P_km_up_value'] += particle['velocity']['P_km_up_value']


        best_individual_pair = generations_data[-1].copy() # Take the best from the last generation
        best_individual_pair['generation'] = j + 1 #Actually it is last generation number, but for consistency let's keep j+1
        best_individual_pair['type'] = 'PSO'
        best_individual_pair['bs_index'] = l
        best_individual_pair['uav_index'] = k
        all_best_individuals_bs.append(best_individual_pair)


        all_best_combinations.append({
            'bs_index': l,
            'uav_index': k,
            'best_fitness': global_best_fitness,
            'best_individual': best_individual_pair,
            'generation_fitness': [gen['fitness'] for gen in generations_data]
        })
        # print(f"Best Fitness for BS {l}, UAV {k}: {global_best_fitness:.4f}")

    # Find best individual for current BS across all UAVs
    best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
    # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


# Select the best unique Base station and UAV-IRS pair using Auction based method
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
                if combination['best_individual']['fitness'] < best_fitness_for_bs:
                    best_fitness_for_bs = combination['best_individual']['fitness']
                    best_combination_for_bs = combination

        if best_combination_for_bs:
            if best_combination_overall is None or best_combination_for_bs['best_individual']['fitness'] < (best_assignments[-1]['best_individual']['fitness'] if best_assignments else float('inf')): # Corrected condition here
                best_combination_overall = best_combination_for_bs

    if best_combination_overall:
        best_assignments.append(best_combination_overall)
        unassigned_bs.remove(best_combination_overall['bs_index'])
        unassigned_uavs.remove(best_combination_overall['uav_index'])

# Print and Plotting
print("\n--- Best Unique UAV Assignments (Auction Based Method with PSO) ---")
best_pair_for_plot = None
min_fitness_for_plot = float('inf')

for assignment in best_assignments:
    print(f"\nBest Assignment for BS {assignment['bs_index']}:")
    print(f"  UAV Index: {assignment['uav_index']}")
    best_ind = assignment['best_individual']
    print(f"  Best Individual:")
    print(f"   Generation: {best_ind['generation']}, Type: {best_ind['type']}")
    print(f"   Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
    for key, value in best_ind['data'].items():
        print(f"   {key}: {value:.4f}")
    print("-" * 20)

    if assignment['best_individual']['fitness'] < min_fitness_for_plot:
        min_fitness_for_plot = assignment['best_individual']['fitness']
        best_pair_for_plot = assignment

# Unassigned BS/UAVs
if unassigned_bs:
    print("\n--- Base Stations without Assigned UAVs ---")
    for bs_index in unassigned_bs:
        print(f"  BS {bs_index} : No UAV is assigned")
        print("-" * 20)

if unassigned_uavs:
    print("\n--- UAVs without Assigned Base Stations ---")
    for uav_index in unassigned_uavs:
        print(f"  UAV {uav_index} : No BS is assigned")
        print("-" * 20)

# Plotting graphs
if best_pair_for_plot:
    fitness_history_best_pair = best_pair_for_plot['generation_fitness']
    generations = range(1, num_generation + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_history_best_pair, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title(f'Fitness Improvement Over Iterations for Best BS-UAV Pair (BS {best_pair_for_plot['bs_index']}, UAV {best_pair_for_plot['uav_index']})')
    plt.grid(True)
    plt.xticks(generations)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo best pair found for plotting.")

# Sum of Best Fitness Values Across Generations
best_pair_combination = min(all_best_combinations, key=lambda x: x['best_fitness'])
best_bs_index_plot = best_pair_combination['bs_index']
best_uav_index_plot = best_pair_combination['uav_index']

generations = []
fitness_values = []

sum_fitness_per_generation = [0] * num_generation
for gen_idx in range(num_generation):
    generation_sum = 0
    for combination in all_best_combinations:
        generation_sum += combination['generation_fitness'][gen_idx]
    sum_fitness_per_generation[gen_idx] = generation_sum

generation_indices = list(range(1, num_generation + 1))

plt.figure(figsize=(10, 6))
plt.plot(generation_indices, sum_fitness_per_generation, marker='o', linestyle='-')
plt.title('Sum of Best Fitness Values Across Iterations (PSO)')
plt.xlabel('Iteration Number')
plt.ylabel('Sum of Best Fitness Values')
plt.grid(True)
plt.xticks(generation_indices)
plt.show()