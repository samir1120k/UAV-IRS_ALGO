# coding: utf-8
#update the previous algorithm to iterate over num_irs_ele
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

Angle_df=pd.read_csv(r'Angle.csv') # number of IRS is 50 store in each column
h_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 50 store in each column
h_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column

Angle_UP_df=pd.read_csv(r'Angle.csv') # number of IRS is 50 store in each column
g_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 50 store in each column
g_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column
Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 50 store in each column
f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 50 store in each column
f_l_m_df=pd.read_csv(r'h_l_m2.csv') # number of IRS is 50 store in each column
f_km1=pd.read_csv(r'f_km.csv')

# Constants
Wl = 35.28
H = 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
# T_ml_down = base['T_ml_down']
# T_km_com = people['T_km_com']
f_km=f_km1['0']
# T_km_up = people['T_km_up']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly = 100

P_km_up=IRS_UP['P_km_up']
# Angle1_col=IRS_UP['Angle'] # number of irs element is 50
# g_l_km_col=IRS_UP['h_l_km'] # number of irs elemnt is 50
# g_l_m_col=IRS_UP['h_l_m'] # number of irs element is 50
p_max=10
p_km_max=10
T_m=10


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
eta=10
kappa=0.5
num_population=50
num_generation = 10 # Number of generations # Fixed generation number

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
    temp1=np.min(h_ml_worst*P_m_down) # Consider if min is the correct aggregation
    if (1+temp1) <= 0:
        return 0  # Return 0 if log argument is non-positive to avoid error
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
                                    # the value for each itaration

def calculate_exp_i_theta(theta): # part of equation 8
  return cmath.exp(1j * theta)
 # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km, num_irs_ele): # part of equation 8, added num_irs_ele
    result=[]
    for i in range(num_irs_ele): # Iterate up to num_irs_ele
        theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
    h_l_m_np = h_l_m.iloc[:num_irs_ele].to_numpy() # Convert Series to numpy array and slice
    h_l_km_np = h_l_km.iloc[:num_irs_ele].to_numpy() # Convert Series to numpy array and slice
    if h_l_m_np.ndim == 1:
        h_l_m_np = h_l_m_np.reshape(1, -1) # Reshape to 2D if necessary
    if h_l_km_np.ndim == 1:
        h_l_km_np = h_l_km_np.reshape(-1, 1) # Reshape to 2D if necessary


    a=np.dot(h_l_m_np,diagonal) # Use numpy arrays for dot product
    b=np.dot(a,h_l_km_np)      # Use numpy arrays for dot product
    final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
    return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    return B*math.log2(1+temp1)
#this is inside the equation 4 have to take summation of h_i_up and P_i_up
def sub(P_i_up,h_il_up):
    return P_i_up*h_il_up


def E_km_com(f_km,T_km_com):
    return eta*(10**(-28))*(f_km**3)*T_km_com

def E_kml_up(P_km_up,T_km_up):
    return P_km_up*T_km_up

def E_kml_har(P_m_har,T_m_har,h_km_har):
    return kappa*P_m_har*T_m_har*h_km_har


#_______________________________________________________________________________________________
# Genetic Algorithm Parameters
num_bs = 5
num_uav_irs = 8

population_size = 50 # Population size for GA
all_best_combinations = []
all_best_individuals = []

# Values for num_irs_ele to iterate over
num_irs_ele_values = range(10, 51, 5) # Iterate from 10 to 50 with step 5
fitness_results_vs_irs = {} # Store results for plotting
best_overall_fitness_irs = float('inf') # Initialize best overall fitness for IRS

for num_irs_ele in num_irs_ele_values:
    all_best_combinations_irs = [] # Store combinations for each num_irs_ele
    print(f"\n--- Starting GA for num_irs_ele = {num_irs_ele} ---")

    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            for i in range(num_population):
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :], num_irs_ele) # Pass Series, and num_irs_ele
                Sub_value+=sub(P_km_up[i],h_il_up_value)

            # Initialize population
            for i in range(population_size):
                f_km_value = f_km[i]
                P_km_up_value = P_km_up.values[i]

                Angle_row = Angle_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice
                h_l_m_row = h_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice
                h_l_km_row = h_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice
                Angle1_row = Angle_UP_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice
                g_l_m_row = g_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice
                g_l_km_row = g_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series and slice


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
                T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row, num_irs_ele) # Pass Series and num_irs_ele
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row, num_irs_ele) # Pass Series, using same function, might need different one if logic is different

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Store initial population data
                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'T_ml_down_value': T_ml_down_value,
                        'f_km_value': f_km_value,
                        'T_km_up_value': T_km_up_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'h_kml_down_value':h_kml_down_value,
                        'T_km_com_value':T_km_com_value
                        }
                })

            generations_data = []
            for j in range(num_generation):
                child_population = []
                for i in range(0, population_size, 2): # Loop through population with step of 2
                    # Crossover
                    parent1 = population[i]
                    parent2 = population[i+1]
                    child_data = {}
                    for key in parent1['data']:
                        child_data[key] = parent1['data'][key] * 0.6 + parent2['data'][key] * (1 - 0.6)

                    # Mutation
                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5
                    if u < P_mutation:
                        for key in child_data:
                            child_data[key] += random.normal(loc=0, scale=1, size=(1))[0]

                    # Compute child fitness
                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        T_ml_down_value = data['T_ml_down_value']
                        f_km_value = data['f_km_value']
                        T_km_up_value = data['T_km_up_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        h_kml_down_value=data['h_kml_down_value']
                        T_km_com_value=data['T_km_com_value']

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
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                        E_ml_har_value = P_m_har_value * T_m_har_value
                        Angle_row_compute = Angle_df.iloc[i, :].iloc[:num_irs_ele]  # Slice for compute function
                        h_l_m_row_compute = h_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        h_l_km_row_compute = h_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                        Angle1_row_compute = Angle_har_df.iloc[i, :].iloc[:num_irs_ele]  # Use same angle row as parent - or you can introduce angle variation here if needed in GA
                        f_l_m_row_compute = f_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        f_l_km_row_compute = f_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                        h_kml_down_value_compute=h_kml_down(Angle_row_compute,h_l_m_row_compute,h_l_km_row_compute, num_irs_ele) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=Dm/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value
                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute=h_kml_down(Angle1_row_compute,f_l_m_row_compute,f_l_km_row_compute,num_irs_ele) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute)
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                                            'P_m_down_value': P_m_down_value,
                                            'P_m_har_value': P_m_har_value,
                                            'T_m_har_value': T_m_har_value,
                                            'T_ml_down_value': T_ml_down_value,
                                            'f_km_value': f_km_value,
                                            'T_km_up_value': T_km_up_value,
                                            'V_lm_vfly_value': V_lm_vfly_value,
                                            'V_lm_hfly_value': V_lm_hfly_value,
                                            'P_km_up_value':P_km_up_value,
                                            'h_kml_down_value':h_kml_down_value_compute, # Use compute value here
                                            'T_km_com_value':T_km_com_value
                                            }
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0 and V_lm_vfly_value>0 and E_kml_har_value>=(E_kml_up_value+E_kml_com_value):
                            return fitness_value, current_data
                        else:
                            return  float('inf'),{} # Return empty dict instead of float('inf') for data

                    child_fitness, child_data1 = compute_fitness(child_data)
                    child_population.append({'fitness': child_fitness, 'data': child_data1})

                # Create new population
                new_population = population + child_population
                new_population = sorted(new_population, key=lambda x: x['fitness'])
                population = new_population[:population_size]
                generations_data.append(population[0].copy())
                # print(population[0])

            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations_irs.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data]
            })
            # print(f"Best Fitness for BS {l}, UAV {k}: {population[0]['fitness']:.4f}")

        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")

    # Auction-based assignment
    combination_lookup = {}
    for combination in all_best_combinations_irs: # Use combinations for current IRS count
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

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

    # Calculate sum of best fitness values in the final generation for current num_irs_ele
    sum_best_fitness_final_gen = 0
    for assignment in best_assignments:
        sum_best_fitness_final_gen += assignment['best_fitness'] # Use best_fitness directly which is from final generation

    # Compare with best_overall_fitness_irs and update if better
    if sum_best_fitness_final_gen < best_overall_fitness_irs:
        best_overall_fitness_irs = sum_best_fitness_final_gen
        fitness_results_vs_irs[num_irs_ele] = best_overall_fitness_irs # Store the new best
    else:
        fitness_results_vs_irs[num_irs_ele] = best_overall_fitness_irs # Store the previous best (or initial if no improvement)


    print(f"\n--- Completed GA for num_irs_ele = {num_irs_ele} ---")
    print(f"Sum of best fitness values in final generation (num_irs_ele={num_irs_ele}): {sum_best_fitness_final_gen:.4f}")
    print(f"Current Best Overall Fitness (num_irs_ele={num_irs_ele}): {best_overall_fitness_irs:.4f}") # Print current best overall
    all_best_combinations.extend(all_best_combinations_irs) # Accumulate all combinations for potential further overall analysis if needed


# Plotting graph for num_irs_ele vs Sum of Best Fitness
num_irs_elements_plot = list(fitness_results_vs_irs.keys())
fitness_values_plot = list(fitness_results_vs_irs.values())

plt.figure(figsize=(10, 6))
plt.plot(num_irs_elements_plot, fitness_values_plot, marker='o', linestyle='-')
plt.xlabel('Number of IRS Elements (num_irs_ele)')
plt.ylabel('Best Sum of Fitness Values (Maintained Best)')
plt.title('Impact of Number of IRS Elements on Best Sum of Fitness')
plt.grid(True)
plt.xticks(num_irs_elements_plot)
plt.tight_layout()
plt.show()