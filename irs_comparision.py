# coding: utf-8
#update the previous algorithm into Hill Climb Search with Debugging and Corrected Neighbor Generation
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

Angle_UP_df=pd.read_csv(r'Angle1.csv') # number of IRS is 50 store in each column
g_l_km_df=pd.read_csv(r'h_l_km1.csv') # number of IRS is 50 store in each column
g_l_m_df=pd.read_csv(r'h_l_m1.csv') # number of IRS is 50 store in each column

Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 50 store in each column
f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 50 store in each column
f_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column
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
eta=10
kappa=0.5

P_km_up=IRS_UP['P_km_up']
# Angle1_col=IRS_UP['Angle'] # number of irs element is 50
# g_l_km_col=IRS_UP['h_l_km1'] # number of irs elemnt is 50 # Corrected column name
# g_l_m_col=IRS_UP['h_l_m1'] # number of irs element is 50 # Corrected column name
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

num_population=50

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
# Hill Climb Search Parameters
num_bs = 5
num_irs_ele=50
num_generation = 10 # Number of generations
num_uav_irs = 8
population_size = 50 # Population size for initialization (used to pick initial solution and neighbors)
all_best_combinations = []
all_best_individuals = []
NEIGHBORHOOD_SIZE = 10 # Number of neighbors to explore in each iteration


# Main Hill Climb Search Loop
# Values for num_irs_ele to iterate over
num_irs_ele_values = range(15, 51, 5) # Iterate from 10 to 50 with step 10
fitness_results_vs_irs = {} # Store results for plotting - CHANGED TO DICT

for num_irs_ele in num_irs_ele_values:
    print(f"\n--- Starting GA for num_irs_ele = {num_irs_ele} ---")
    all_best_combinations_irs = [] # Store combinations for each num_irs_ele

    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = [] # Still use population list, but for initialization
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            for i in range(num_population):
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)

            # Initialize population (same as RS to keep functionality consistent and for initial solution)
            initial_population_data = [] # Store data for picking initial solution and neighbors
            for i in range(population_size):
                f_km_value = f_km[i]
                P_km_up_value = P_km_up.values[i]

                Angle_row = Angle_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                h_l_m_row = h_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                h_l_km_row = h_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                Angle1_row = Angle_UP_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_m_row = g_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_km_row = g_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series


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
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

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
                initial_population_data.append({
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

            # Select initial solution for Hill Climbing (e.g., best from initial population)
            current_individual_entry = min(initial_population_data, key=lambda x: x['fitness'])
            current_individual = current_individual_entry['data']
            best_fitness = current_individual_entry['fitness']
            best_individual = current_individual

            generations_data = []
            previous_best_fitness = best_fitness # Keep track of the best fitness *before* this generation
            for j in range(num_generation):
                # Hill Climbing - Generate neighbors by perturbing the current individual's parameters
                neighbors = []
                for _ in range(NEIGHBORHOOD_SIZE):
                    neighbor_data = current_individual.copy()
                    # Perturb each parameter with a small random value (e.g., +/- 5%)
                    perturbation_factor = 0.05

                    neighbor_data['f_km_value'] *= (1 + random.uniform(-perturbation_factor, perturbation_factor))
                    neighbor_data['P_km_up_value'] *= (1 + random.uniform(-perturbation_factor, perturbation_factor))
                    neighbor_data['V_lm_vfly_value'] *= (1 + random.uniform(-perturbation_factor, perturbation_factor))
                    neighbor_data['V_lm_hfly_value'] *= (1 + random.uniform(-perturbation_factor, perturbation_factor))
                    # Add perturbations for other relevant parameters if needed, e.g., power values.
                    neighbors.append({'data': neighbor_data})

                best_neighbor_fitness = float('inf')
                best_neighbor_data = {}

                for neighbor_entry in neighbors:
                    neighbor_data = neighbor_entry['data']

                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        T_ml_down_value = data['T_ml_down_value'] # using current_individual's T_ml_down_value, might need recalculation if P_m_down changes significantly in neighbor
                        f_km_value = data['f_km_value']
                        T_km_up_value = data['T_km_up_value'] # using current_individual's T_km_up_value, might need recalculation if P_km_up changes significantly in neighbor
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        h_kml_down_value=data['h_kml_down_value'] # using current_individual's h_kml_down_value and angle, assuming angle doesn't change in neighbor generation for simplicity now.
                        T_km_com_value=data['T_km_com_value'] # using current_individual's T_km_com_value, might need recalculation if f_km changes significantly in neighbor

                        # Calculate Bh and p_l_b for neighbor
                        Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                        Bh = max(1, Bh)
                        p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

                        # Calculate time and energy values - Recalculate dependent values based on perturbed parameters
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
                        E_ml_har_value = P_m_har_value * T_m_har_value


                        Angle_row_compute = Angle_df.iloc[i, :].iloc[:num_irs_ele]  #  Still using 'i' from outer loop initialization - POTENTIAL ISSUE - Should ideally relate to current BS/UAV
                        h_l_m_row_compute = h_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        h_l_km_row_compute = h_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                        Angle1_row_compute = Angle_har_df.iloc[i, :].iloc[:num_irs_ele]
                        f_l_m_row_compute = f_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        f_l_km_row_compute = f_l_km_df.iloc[i, :].iloc[:num_irs_ele]


                        h_kml_down_value_compute=h_kml_down(Angle_row_compute,h_l_m_row_compute,h_l_km_row_compute)
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0:
                            R_ml_down_value = 1e-9
                        T_ml_down_value=Dm/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value


                        T_km_com_value = D_km / f_km_value # Recalculate T_km_com

                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value) # T_km_up and T_ml_down might need recalculation based on perturbed values
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                        h_kml_har_value_compute=h_kml_down(Angle1_row_compute,f_l_m_row_compute,f_l_km_row_compute)
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute)
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                        if V_lm_hfly_value<0 or T_m_har_value<0 or T_ml_down_value<0 or T_km_up_value<0 or P_m_har_value>p_max or P_m_down_value>p_max or P_km_up_value>p_km_max or (T_km_com_value+T_km_up_value+T_ml_down_value)>T_m or f_km_value<0 or V_lm_vfly_value<0 or E_kml_har_value<(E_kml_up_value+E_kml_com_value):
                            return float('inf'),{}
                        else:
                            return fitness_value, data


                    current_neighbor_fitness, current_neighbor_data = compute_fitness(neighbor_data)


                    if current_neighbor_fitness < best_neighbor_fitness:
                        best_neighbor_fitness = current_neighbor_fitness
                        best_neighbor_data = neighbor_data


                # print(f"Generation {j+1}: Current Fitness = {current_individual_entry['fitness']:.4f}, Best Neighbor Fitness = {best_neighbor_fitness:.4f}") # Debug print

                if best_neighbor_fitness < current_individual_entry['fitness']:
                    current_individual = best_neighbor_data.copy() # Move to the better neighbor
                    current_individual_entry['fitness'] = best_neighbor_fitness # Update current individual's fitness in entry
                    best_fitness = best_neighbor_fitness
                    best_individual = current_individual
                    previous_best_fitness = best_neighbor_fitness

                    # print(f"  Improved Fitness to: {best_fitness:.4f}") # Debug print

                # else:
                #     print("  No improvement in this generation.") # Debug print/


                generations_data.append({'fitness': current_individual_entry['fitness'], 'data': current_individual}) # Store data for each generation


            best_individual_pair = {'fitness': best_fitness, 'data': best_individual} # Reconstruct the pair structure
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'HC' # Indicate Hill Climb Search type
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': best_fitness,
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data]
            })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])


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

    # Print and Plotting
    print("\n--- Best Unique UAV Assignments (Auction Based Method) - Hill Climb Search ---") # Modified print statement
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')
    sum_best_fitness_current_irs = 0 # Initialize sum for current num_irs_ele

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f"  UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f"  Best Individual:")
        print(f"    Generation: {best_ind['generation']}, Type: {best_ind['type']}") # Indicate HC type
        print(f"    Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        for key, value in best_ind['data'].items():
            print(f"    {key}: {value:.4f}")
        print("-" * 20)

        sum_best_fitness_current_irs += assignment['best_fitness'] # Sum up best fitnesses

        if assignment['best_individual']['fitness'] < min_fitness_for_plot and assignment['best_individual']['fitness'] != float('inf'): # Added condition to exclude inf
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    # Store sum of best fitness for the current num_irs_ele
    fitness_results_vs_irs[num_irs_ele] = sum_best_fitness_current_irs

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


# Plotting graph for num_irs_ele vs Sum of Best Fitness
num_irs_elements_plot = list(fitness_results_vs_irs.keys()) # Use keys of dict for num_irs_ele
fitness_values_plot_HC = list(fitness_results_vs_irs.values()) # Use values of dict for sum of fitness

# plt.figure(figsize=(10, 6))
# plt.plot(num_irs_elements_plot, fitness_values_plot, marker='o', linestyle='-')
# plt.xlabel('Number of IRS Elements (num_irs_ele)')
# plt.ylabel('Sum of Best Fitness Values (Auctioned Assignments)') # Updated Y-axis label
# plt.title('Impact of Number of IRS Elements on Sum of Best Fitness') # Updated title
# plt.grid(True)
# plt.xticks(num_irs_elements_plot)
# plt.tight_layout()
# plt.show()

#____________________________________________________________________________________________________

#Genetic Algorithm
# coding: utf-8
#update the previous algorithm
# import pandas as pd
# import numpy as np
# import random
# import math
# from numpy import random
# import cmath
# import matplotlib.pyplot as plt


# # Load datasets related to Base Stations, UAVs, and Clients
# base = pd.read_csv(r'BS_data.csv')
# uav = pd.read_csv(r'UAV_data.csv')
# people = pd.read_csv(r'people_data.csv')
# IRS=pd.read_csv(r'IRS_data.csv')
# IRS_UP=pd.read_csv(r'IRS_data_up.csv')

# Angle_df=pd.read_csv(r'Angle.csv') # number of IRS is 50 store in each column
# h_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 50 store in each column
# h_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column

# Angle_UP_df=pd.read_csv(r'Angle.csv') # number of IRS is 50 store in each column
# g_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 50 store in each column
# g_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column

# Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 50 store in each column
# f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 50 store in each column
# f_l_m_df=pd.read_csv(r'h_l_m2.csv') # number of IRS is 50 store in each column


# f_km1=pd.read_csv(r'f_km.csv')

# # Constants
# Wl = 35.28
# H = 20
# P_m_har = base['P_m_har']
# T_m_har = base['T_m_har']
# P_m_down = base['P_m_down']
# # T_ml_down = base['T_ml_down']
# # T_km_com = people['T_km_com']
# f_km=f_km1['0']
# # T_km_up = people['T_km_up']
# V_lm_vfly = uav['V_lm_vfly']
# V_lm_hfly = uav['V_lm_hfly']
# D_l_hfly = 100

# P_km_up=IRS_UP['P_km_up']
# # Angle1_col=IRS_UP['Angle'] # number of irs element is 50
# # g_l_km_col=IRS_UP['h_l_km'] # number of irs elemnt is 50
# # g_l_m_col=IRS_UP['h_l_m'] # number of irs element is 50
# p_max=10 # moved inside loop
# p_km_max=10
# T_m=10
# T_m_values = range(5, 11) # T_m values from 1


# # Additional constants for calculations
# delta = 2
# Ar = 0.1256
# s = 0.05
# Nr = 4
# V_tip = 102
# Cd = 0.022
# Af = 0.2113
# D_km = 0.5
# Dm=0.49
# B=10 #MHz
# sigma_km=10**(-13)
# eta=10
# kappa=0.5
# num_population=50

# # Fitness function to calculate total energy consumption
# def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
#     return E_ml_har + E_ml_down + E_ml_UAV

# # Energy consumption of the UAV-IRS
# def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
#     return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# # Power calculations for different flight modes
# def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
#     temp2 = Nr * Bh * Ar
#     temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
#     return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

# def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
#     return P_lm_blade + P_lm_fuselage + P_lm_induced

# def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
#     return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

# def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
#     return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

# def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
#     return Wl * (np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2)**(1 / 2))

# def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
#     temp1 = Nr * P_l_b
#     temp2 = abs(2 * (Nr * Bh * Ar))
#     temp3 = np.sqrt(temp2)
#     temp4 = (Wl**3 / 2) / temp3
#     return temp1 + temp4

# def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
#     return T_km_com + T_kml_up + T_ml_down

# def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
#     temp1=np.min(h_ml_worst*P_m_down) # Consider if min is the correct aggregation
#     if (1+temp1) <= 0:
#         return 0  # Return 0 if log argument is non-positive to avoid error
#     return B*math.log2(1+temp1)

# def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
#     return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
#                 # the value for each itaration

# def calculate_exp_i_theta(theta): # part of equation 8
#   return cmath.exp(1j * theta)
#  # 1j represents the imaginary unit in Python

# def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8
#     result=[]
#     for i in range(len(Angle)):
#         theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
#         results= calculate_exp_i_theta(theta_radians)
#         result.append(results)

#     diagonal=np.diag(result)
#     # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
#     h_l_m_np = h_l_m.to_numpy() # Convert Series to numpy array
#     h_l_km_np = h_l_km.to_numpy() # Convert Series to numpy array
#     if h_l_m_np.ndim == 1:
#         h_l_m_np = h_l_m_np.reshape(1, -1) # Reshape to 2D if necessary
#     if h_l_km_np.ndim == 1:
#         h_l_km_np = h_l_km_np.reshape(-1, 1) # Reshape to 2D if necessary


#     a=np.dot(h_l_m_np,diagonal) # Use numpy arrays for dot product
#     b=np.dot(a,h_l_km_np)      # Use numpy arrays for dot product
#     final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
#     return (final**2)

# def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
#     temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
#     return B*math.log2(1+temp1)
# #this is inside the equation 4 have to take summation of h_i_up and P_i_up
# def sub(P_i_up,h_il_up):
#     return P_i_up*h_il_up

# def E_km_com(f_km,T_km_com):
#     return eta*(10**(-28))*(f_km**3)*T_km_com

# def E_kml_up(P_km_up,T_km_up):
#     return P_km_up*T_km_up

# def E_kml_har(P_m_har,T_m_har,h_km_har):
#     return kappa*P_m_har*T_m_har*h_km_har

#_______________________________________________________________________________________________
# Genetic Algorithm Parameters
num_bs = 5
num_irs_ele=50
num_generation = 10 # Number of generations
num_uav_irs = 8
population_size = 50 # Population size for GA



all_best_combinations = []
all_best_individuals = []
# Main Hill Climb Search Loop
# Values for num_irs_ele to iterate over
num_irs_ele_values = range(15, 51, 5) # Iterate from 10 to 50 with step 10
fitness_results_vs_irs = {} # Store results for plotting - CHANGED TO DICT

for num_irs_ele in num_irs_ele_values:
    print(f"\n--- Starting GA for num_irs_ele = {num_irs_ele} ---")
    all_best_combinations_irs = [] # Store combinations for each num_irs_ele

    # Main Genetic Algorithm Loop
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
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)

            # Initialize population
            for i in range(population_size):
                f_km_value = f_km[i]
                P_km_up_value = P_km_up.values[i]

                Angle_row = Angle_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                h_l_m_row = h_l_m_df.iloc[i, :].iloc[:num_irs_ele]# Get row as Series
                h_l_km_row = h_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                Angle1_row = Angle_UP_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_m_row = g_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_km_row = g_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series


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
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

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
                            child_data[key] += random.normal(loc=0, scale=0.1, size=(1))[0]

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
                        Angle_row_compute = Angle_df.iloc[i, :].iloc[:num_irs_ele]  # Use same angle row as parent - or you can introduce angle variation here if needed in GA
                        h_l_m_row_compute = h_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        h_l_km_row_compute = h_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                        Angle1_row_compute = Angle_har_df.iloc[i, :].iloc[:num_irs_ele]  # Use same angle row as parent - or you can introduce angle variation here if needed in GA
                        f_l_m_row_compute = f_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                        f_l_km_row_compute = f_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                        h_kml_down_value_compute=h_kml_down(Angle_row_compute,h_l_m_row_compute,h_l_km_row_compute) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
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
                        h_kml_har_value_compute=h_kml_down(Angle1_row_compute,f_l_m_row_compute,f_l_km_row_compute) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
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
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
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

            all_best_combinations.append({
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

    # Print and Plotting
    # print(f"\n--- Best Unique UAV Assignments (Auction Based Method) for p_max = {p_max} ---")
    # best_pair_for_plot = None
    # min_fitness_for_plot = float('inf')

    sum_fitness_current_p_max = 0 # Sum of best fitness for current p_max
    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f"  UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f"  Best Individual:")
        print(f"   Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"   Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        for key, value in best_ind['data'].items():
            print(f"    {key}: {value:.4f}")
        print("-" * 20)
        sum_fitness_current_p_max += best_ind['fitness'] # Sum fitness values

        if assignment['best_individual']['fitness'] < min_fitness_for_plot:
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    all_best_combinations_irs.append(sum_fitness_current_p_max) # Store sum of fitness for this p_max

    # Store results for plotting outside the p_max loop
    fitness_results_vs_irs[num_irs_ele] = sum_fitness_current_p_max


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

# Plotting graph for Sum of Fitness vs Number of IRS Elements
# print("\nFitness results vs IRS Elements:", fitness_results_vs_irs) # Print the dictionary to verify data

num_irs_elements_plot = list(fitness_results_vs_irs.keys()) # Use keys of dict for num_irs_ele
fitness_values_plot_GA = list(fitness_results_vs_irs.values()) # Use values of dict for sum of fitness

# plt.figure(figsize=(10, 6))
# plt.plot(num_irs_elements_plot, fitness_values_plot, marker='o', linestyle='-')
# plt.xlabel('Number of IRS Elements (num_irs_ele)')
# plt.ylabel('Sum of Best Fitness Values (Auctioned Assignments)') # Updated Y-axis label
# plt.title('Impact of Number of IRS Elements on Sum of Best Fitness') # Updated title
# plt.grid(True)
# plt.xticks(num_irs_elements_plot)
# plt.tight_layout()
# plt.show()

#_____________________________________________________________________________________________
#Random Search

# coding: utf-8
# #update the previous algorithm into Random Search
# import pandas as pd
# import numpy as np
# import random
# import math
# from numpy import random
# import cmath
# import matplotlib.pyplot as plt


# # Load datasets related to Base Stations, UAVs, and Clients
# base = pd.read_csv(r'BS_data.csv')
# uav = pd.read_csv(r'UAV_data.csv')
# people = pd.read_csv(r'people_data.csv')
# IRS=pd.read_csv(r'IRS_data.csv')
# IRS_UP=pd.read_csv(r'IRS_data_up.csv')

# Angle_df=pd.read_csv(r'Angle.csv') # number of IRS is 50 store in each column
# h_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 50 store in each column
# h_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column

# Angle_UP_df=pd.read_csv(r'Angle1.csv') # number of IRS is 50 store in each column
# g_l_km_df=pd.read_csv(r'h_l_km1.csv') # number of IRS is 50 store in each column
# g_l_m_df=pd.read_csv(r'h_l_m1.csv') # number of IRS is 50 store in each column

# Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 50 store in each column
# f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 50 store in each column
# f_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 50 store in each column
# f_km1=pd.read_csv(r'f_km.csv')

# # Constants
# Wl = 35.28
# H = 20
# P_m_har = base['P_m_har']
# T_m_har = base['T_m_har']
# P_m_down = base['P_m_down']
# # T_ml_down = base['T_ml_down']
# # T_km_com = people['T_km_com']
# f_km=f_km1['0']
# # T_km_up = people['T_km_up']
# V_lm_vfly = uav['V_lm_vfly']
# V_lm_hfly = uav['V_lm_hfly']
# D_l_hfly = 100
# eta=10
# kappa=0.5

# P_km_up=IRS_UP['P_km_up']
# # Angle1_col=IRS_UP['Angle'] # number of irs element is 50
# # g_l_km_col=IRS_UP['h_l_km'] # number of irs elemnt is 50
# # g_l_m_col=IRS_UP['h_l_m'] # number of irs element is 50
# p_km_max=10
# T_m=10
# p_max=10


# # Additional constants for calculations
# delta = 2
# Ar = 0.1256
# s = 0.05
# Nr = 4
# V_tip = 102
# Cd = 0.022
# Af = 0.2113
# D_km = 0.5
# Dm=0.49
# B=10 #MHz
# sigma_km=10**(-13)

# num_population=50

# # Fitness function to calculate total energy consumption
# def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
#     return E_ml_har + E_ml_down + E_ml_UAV

# # Energy consumption of the UAV-IRS
# def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
#     return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# # Power calculations for different flight modes
# def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
#     temp2 = Nr * Bh * Ar
#     temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
#     return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

# def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
#     return P_lm_blade + P_lm_fuselage + P_lm_induced

# def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
#     return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

# def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
#     return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

# def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
#     return Wl * (np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2)**(1 / 2))

# def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
#     temp1 = Nr * P_l_b
#     temp2 = abs(2 * (Nr * Bh * Ar))
#     temp3 = np.sqrt(temp2)
#     temp4 = (Wl**3 / 2) / temp3
#     return temp1 + temp4

# def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
#     return T_km_com + T_kml_up + T_ml_down

# def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
#     temp1=np.min(h_ml_worst*P_m_down) # Consider if min is the correct aggregation
#     if (1+temp1) <= 0:
#         return 0  # Return 0 if log argument is non-positive to avoid error
#     return B*math.log2(1+temp1)

# def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
#     return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
#                                         # the value for each itaration

# def calculate_exp_i_theta(theta): # part of equation 8
#   return cmath.exp(1j * theta)
#  # 1j represents the imaginary unit in Python

# def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8
#     result=[]
#     for i in range(len(Angle)):
#         theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
#         results= calculate_exp_i_theta(theta_radians)
#         result.append(results)

#     diagonal=np.diag(result)
#     # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
#     h_l_m_np = h_l_m.to_numpy() # Convert Series to numpy array
#     h_l_km_np = h_l_km.to_numpy() # Convert Series to numpy array
#     if h_l_m_np.ndim == 1:
#         h_l_m_np = h_l_m_np.reshape(1, -1) # Reshape to 2D if necessary
#     if h_l_km_np.ndim == 1:
#         h_l_km_np = h_l_km_np.reshape(-1, 1) # Reshape to 2D if necessary


#     a=np.dot(h_l_m_np,diagonal) # Use numpy arrays for dot product
#     b=np.dot(a,h_l_km_np)       # Use numpy arrays for dot product
#     final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
#     return (final**2)

# def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
#     temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
#     return B*math.log2(1+temp1)
# #this is inside the equation 4 have to take summation of h_i_up and P_i_up
# def sub(P_i_up,h_il_up):
#     return P_i_up*h_il_up

# def E_km_com(f_km,T_km_com):
#     return eta*(10**(-28))*(f_km**3)*T_km_com

# def E_kml_up(P_km_up,T_km_up):
#     return P_km_up*T_km_up

# def E_kml_har(P_m_har,T_m_har,h_km_har):
#     return kappa*P_m_har*T_m_har*h_km_har

#_______________________________________________________________________________________________
# Random Search Parameters
num_bs = 5
num_irs_ele=50
num_generation = 10 # Number of generations
num_uav_irs = 8
population_size = 50 # Population size for RS (will be used for random selection)


# Parameter sweep for p_max


all_best_combinations = []
all_best_individuals = []
num_irs_ele_values = range(15, 51, 5) # Iterate from 10 to 50 with step 10
fitness_results_vs_irs = {} # Store results for plotting - CHANGED TO DICT

for num_irs_ele in num_irs_ele_values:
    print(f"\n--- Starting GA for num_irs_ele = {num_irs_ele} ---")
    all_best_combinations_irs = [] # Store combinations for each num_irs_ele
    sum_fitness_per_p_max_value = 0

    # Main Random Search Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = [] # Still use population list, but for random selection
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            for i in range(num_population):
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)

            # Initialize population (same as GA to keep functionality consistent)
            for i in range(population_size):
                f_km_value = f_km[i]
                P_km_up_value = P_km_up.values[i]

                Angle_row = Angle_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                h_l_m_row = h_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                h_l_km_row = h_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                Angle1_row = Angle_UP_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_m_row = g_l_m_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series
                g_l_km_row = g_l_km_df.iloc[i, :].iloc[:num_irs_ele] # Get row as Series


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
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

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

            #best_fitness_generation = best_fitness # No need for separate generation best anymore
            generations_data = []
            previous_best_fitness = best_fitness # Keep track of the best fitness *before* this generation
            for j in range(num_generation):
                # Random Selection - Instead of GA's selection, crossover, mutation
                random_individual = random.choice(population) # Randomly select from initial population

                # Compute fitness for the randomly selected individual (already computed in initialization, can just use it or re-compute if needed - re-computing for clarity in RS logic)
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
                    Angle_row_compute = Angle_df.iloc[i, :].iloc[:num_irs_ele]  # Use same angle row as parent
                    h_l_m_row_compute = h_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                    h_l_km_row_compute = h_l_km_df.iloc[i, :].iloc[:num_irs_ele]

                    Angle1_row_compute = Angle_har_df.iloc[i, :].iloc[:num_irs_ele]  # Use same angle row as parent
                    f_l_m_row_compute = f_l_m_df.iloc[i, :].iloc[:num_irs_ele]
                    f_l_km_row_compute = f_l_km_df.iloc[i, :].iloc[:num_irs_ele]


                    h_kml_down_value_compute=h_kml_down(Angle_row_compute,h_l_m_row_compute,h_l_km_row_compute) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well
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
                    h_kml_har_value_compute=h_kml_down(Angle1_row_compute,f_l_m_row_compute,f_l_km_row_compute) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well
                    E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute)
                    E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                    E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                    # Calculate fitness
                    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                    if V_lm_hfly_value<0 or T_m_har_value<0 or T_ml_down_value<0 or T_km_up_value<0 or P_m_har_value>p_max or P_m_down_value>p_max or P_km_up_value>p_km_max or (T_km_com_value+T_km_up_value+T_ml_down_value)>T_m or f_km_value<0 or V_lm_vfly_value<0 or E_kml_har_value<(E_kml_up_value+E_kml_com_value):
                        return float('inf'),{}
                    else:
                        return fitness_value, data # Return original data as it's randomly selected


                child_fitness, child_data1 = compute_fitness(random_individual['data']) # Recompute fitness for random individual (or could just use pre-computed if population is not modified)


                # In RS, we simply compare the fitness of the random individual with the current best
                if child_fitness < previous_best_fitness: # Compare with the best fitness from *before* this iteration
                    best_fitness = child_fitness # Update overall best fitness if current is better
                    best_individual = child_data1.copy() # Store data
                    previous_best_fitness = child_fitness # Update previous best to current best
                else:
                    child_fitness = previous_best_fitness # If not better, keep previous best as current fitness for this iteration


                generations_data.append({'fitness': child_fitness, 'data': child_data1}) # Store data for each generation


            best_individual_pair = {'fitness': best_fitness, 'data': best_individual} # Reconstruct the pair structure
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'RS' # Indicate Random Search
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': best_fitness,
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data]
            })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])


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
        # print(f"Start of auction loop: unassigned_bs={unassigned_bs}, unassigned_uavs={unassigned_uavs}") # Debugging print
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

        # print(f"best_combination_overall: {best_combination_overall}") # Debugging print
        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])
            # print(f"Assigned BS {best_combination_overall['bs_index']}, UAV {best_combination_overall['uav_index']}. Remaining unassigned_bs={unassigned_bs}, unassigned_uavs={unassigned_uavs}") # Debugging print
        else:
            print("No best combination found in this auction iteration. Loop might be stuck.") # Debugging print
            break # Add break to prevent infinite loop if no best combination is found


    # Sum of Best Fitness Values Across Iterations (for RS, "generations" are just iterations)
    sum_fitness_per_generation = [0] * num_generation
    for gen_idx in range(num_generation):
        generation_sum = 0
        for combination in all_best_combinations:
            fitness_value = combination['generation_fitness'][gen_idx]
            valid_fitness = combination['generation_fitness'][gen_idx]
            if valid_fitness != float('inf'): # Exclude inf values from sum
                generation_sum += valid_fitness
        sum_fitness_per_generation[gen_idx] = generation_sum

    fitness_results_vs_irs[num_irs_ele] = sum([assignment['best_fitness'] for assignment in best_assignments])


# Plotting graph for num_irs_ele vs Sum of Best Fitness
num_irs_elements_plot = list(fitness_results_vs_irs.keys()) # Use keys of dict for num_irs_ele
fitness_values_plot_RS = list(fitness_results_vs_irs.values()) # Use values of dict for sum of fitness

# plt.figure(figsize=(10, 6))
# plt.plot(num_irs_elements_plot, fitness_values_plot, marker='o', linestyle='-')
# plt.xlabel('Number of IRS Elements (num_irs_ele)')
# plt.ylabel('Sum of Best Fitness Values (Auctioned Assignments)') # Updated Y-axis label
# plt.title('Impact of Number of IRS Elements on Sum of Best Fitness') # Updated title
# plt.grid(True)
# plt.xticks(num_irs_elements_plot)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12, 7))



plt.plot(num_irs_elements_plot, fitness_values_plot_HC,  marker='s', linestyle='--', label='hill climb')
plt.plot(num_irs_elements_plot, fitness_values_plot_GA, marker='*', linestyle='--', label='Genetic Algorithm')
plt.plot(num_irs_elements_plot, fitness_values_plot_RS,marker='o', linestyle='--', label='Random Search') 
plt.xlabel('IRS_ELEMENT')
plt.ylabel('Sum of Best Fitness Values (Auctioned Assignments)')
plt.title('Comparison of Algorithm Performance vs. IRS-ELEMENT')
plt.grid(True)
plt.xticks(num_irs_elements_plot)
plt.legend()
plt.tight_layout()
plt.show()