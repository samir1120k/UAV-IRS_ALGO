# coding: utf-8
#update the previous algorithm to Hill Climb

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
eta=10
kappa=0.5


P_km_up=IRS_UP['P_km_up']
# Angle1_col=IRS_UP['Angle'] # number of irs element is 50
# g_l_km_col=IRS_UP['h_l_km'] # number of irs elemnt is 50
# g_l_m_col=IRS_UP['h_l_m'] # number of irs element is 50
# p_max=10 # p_max is now varying in the outer loop
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

num_population=50 # not using population size in hill climb, but keeping for initial solution

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
# Hill Climb Algorithm Parameters
num_bs = 5
num_irs_ele=50
num_iterations_hc = 30 # Number of iterations for Hill Climbing
num_uav_irs = 8
num_initial_solutions = 50 # Number of initial solutions to try for Hill Climbing
all_best_combinations_hc = []
all_best_individuals_hc = []

fitness_sum_vs_pmax = []
p_max_values = range(1, 11) # p_max from 1 to 10

for p_max in p_max_values:
    print(f"\n--- Running Hill Climb for p_max = {p_max} ---")
    all_best_combinations_hc = [] # Clear for each p_max value
    all_best_individuals_hc = []
    for l in range(num_bs):
        all_best_individuals_bs_hc = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        for k in range(num_uav_irs):
            best_fitness_hc = float('inf')
            best_individual_hc = {}
            best_generation_data_hc = [] # Store fitness per iteration for plotting
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0
            initial_solutions = []

            for i in range(num_initial_solutions): # Generate multiple initial solutions
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[i, :],g_l_km_df.iloc[i, :]) # Pass Series
                Sub_value+=sub(P_km_up[i],h_il_up_value)

            # Initialize initial solution - picking one from the initial set, or you can start with random values
            initial_index = 0 # You can randomize this index if needed
            f_km_value_current = f_km[initial_index]
            P_km_up_value_current = P_km_up.values[initial_index]

            Angle_row_current = Angle_df.iloc[initial_index, :]
            h_l_m_row_current = h_l_m_df.iloc[initial_index, :]
            h_l_km_row_current = h_l_km_df.iloc[initial_index, :]
            Angle1_row_current = Angle_UP_df.iloc[initial_index, :]
            g_l_m_row_current = g_l_m_df.iloc[initial_index, :]
            g_l_km_row_current = g_l_km_df.iloc[initial_index, :]


            current_individual = {
                'f_km_value': f_km_value_current,
                'P_km_up_value': P_km_up_value_current,
                'Angle_row': Angle_row_current,
                'h_l_m_row': h_l_m_row_current,
                'h_l_km_row': h_l_km_row_current,
                'Angle1_row': Angle1_row_current,
                'g_l_m_row': g_l_m_row_current,
                'g_l_km_row': g_l_km_row_current,
                'P_m_down_value': P_m_down_value,
                'P_m_har_value': P_m_har_value,
                'T_m_har_value': T_m_har_value,
                'V_lm_vfly_value': V_lm_vfly_value,
                'V_lm_hfly_value': V_lm_hfly_value
            }

            best_individual_hc_data = {} # Store best individual data
            best_fitness_hc_current = float('inf') # Initialize with infinity

            for j in range(num_iterations_hc):
                # Calculate fitness for current individual
                def calculate_fitness_hc(individual_data):
                    f_km_value_hc = individual_data['f_km_value']
                    P_km_up_value_hc = individual_data['P_km_up_value']
                    Angle_row_hc = individual_data['Angle_row']
                    h_l_m_row_hc = individual_data['h_l_m_row']
                    h_l_km_row_hc = individual_data['h_l_km_row']
                    Angle1_row_hc = individual_data['Angle1_row']
                    g_l_m_row_hc = individual_data['g_l_m_row']
                    g_l_km_row_hc = individual_data['g_l_km_row']
                    P_m_down_value_hc = individual_data['P_m_down_value']
                    P_m_har_value_hc = individual_data['P_m_har_value']
                    T_m_har_value_hc = individual_data['T_m_har_value']
                    V_lm_vfly_value_hc = individual_data['V_lm_vfly_value']
                    V_lm_hfly_value_hc = individual_data['V_lm_hfly_value']


                    # Calculate Bh and p_l_b for current individual
                    Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                    Bh = max(1, Bh)
                    p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                    # Calculate power values
                    P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value_hc)
                    P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value_hc)
                    P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value_hc)

                    # Calculate time and energy values
                    T_l_vfly_value = H_value / V_lm_vfly_value_hc
                    T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value_hc # Corrected: D_l_hfly / V_lm_hfly
                    E_ml_har_value = kappa * P_m_har_value_hc * T_m_har_value_hc # using kappa here
                    h_kml_down_value_hc=h_kml_down(Angle_row_hc,h_l_m_row_hc,h_l_km_row_hc) # Pass Series
                    h_ml_worst_value=h_ml_worst(h_kml_down_value_hc,sigma_km)
                    R_ml_down_value=R_ml_down(B,P_m_down_value_hc,h_ml_worst_value)
                    if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                        R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                    T_ml_down_value=Dm/R_ml_down_value
                    E_ml_down_value = P_m_down_value_hc * T_ml_down_value
                    T_km_com_value = D_km / f_km_value_hc
                    h_kml_up_value=h_kml_down(Angle1_row_hc,g_l_m_row_hc,g_l_km_row_hc) # Pass Series, using same function, might need different one if logic is different

                    R_kml_up_value=R_kml_up(B,P_km_up_value_hc,h_kml_up_value,Sub_value,sigma_km)
                    T_km_up_value=Dm/R_kml_up_value # equation number 5
                    T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                    P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                    P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value_hc, p_l_b, Nr, Ar, Bh)
                    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                    h_kml_har_value_compute=h_kml_down(Angle_har_df.iloc[initial_index, :],f_l_m_df.iloc[initial_index, :],f_l_km_df.iloc[initial_index, :]) # Using initial angle rows
                    E_kml_har_value=E_kml_har(P_m_har_value_hc,T_m_har_value_hc,h_kml_har_value_compute)
                    E_kml_com_value = E_km_com(f_km_value_hc, T_km_com_value)
                    E_kml_up_value=E_kml_up(P_km_up_value_hc,T_km_up_value)


                    # Calculate fitness
                    result_fitness_hc = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                    if V_lm_hfly_value_hc>0 and T_m_har_value_hc>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value_hc<=p_max and P_m_down_value_hc<=p_max and P_km_up_value_hc<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value_hc>0 and V_lm_vfly_value_hc>0 and E_kml_har_value>=(E_kml_up_value+E_kml_com_value):
                        return result_fitness_hc, {
                                                    'P_m_down_value': P_m_down_value_hc,
                                                    'P_m_har_value': P_m_har_value_hc,
                                                    'T_m_har_value': T_m_har_value_hc,
                                                    'T_ml_down_value': T_ml_down_value,
                                                    'f_km_value': f_km_value_hc,
                                                    'T_km_up_value': T_km_up_value,
                                                    'V_lm_vfly_value': V_lm_vfly_value_hc,
                                                    'V_lm_hfly_value': V_lm_hfly_value_hc,
                                                    'P_km_up_value':P_km_up_value_hc,
                                                    'h_kml_down_value':h_kml_down_value_hc,
                                                    'T_km_com_value':T_km_com_value
                                                    }
                    else:
                        return float('inf'), {}


                current_fitness_hc, current_data_hc = calculate_fitness_hc(current_individual)
                best_generation_data_hc.append(current_fitness_hc) # Store fitness for plotting

                if current_fitness_hc < best_fitness_hc_current:
                    best_fitness_hc_current = current_fitness_hc
                    best_individual_hc_data = current_data_hc.copy()


                # Generate neighbors - perturb each parameter slightly
                neighbor_individual = current_individual.copy()
                perturbation_factor = 0.1 # Adjust for step size in search
                params_to_perturb = ['f_km_value', 'P_km_up_value', 'V_lm_vfly_value', 'V_lm_hfly_value'] # List of params to tune

                for param_name in params_to_perturb:
                    # Perturb each parameter and check neighbor
                    original_value = current_individual[param_name]
                    # Try increasing and decreasing the parameter value
                    for direction in [-1, 1]:
                        perturbed_value = original_value * (1 + direction * perturbation_factor)
                        if perturbed_value > 0: # Ensure values are positive where required
                            neighbor_individual[param_name] = perturbed_value

                            neighbor_fitness_hc, neighbor_data_hc = calculate_fitness_hc(neighbor_individual)

                            if neighbor_fitness_hc < current_fitness_hc: # If neighbor is better, move to neighbor
                                current_fitness_hc = neighbor_fitness_hc
                                current_individual = neighbor_individual.copy() # Update current individual
                                best_individual_hc_data = neighbor_data_hc.copy() # Update best individual data if current is updated


            best_individual_hc = {'fitness': current_fitness_hc, 'data': best_individual_hc_data} # Use best fitness found in HC
            best_individual_hc['generation'] = j + 1 # Misleading 'generation' here, actually iteration count
            best_individual_hc['type'] = 'HC'
            best_individual_hc['bs_index'] = l
            best_individual_hc['uav_index'] = k
            all_best_individuals_bs_hc.append(best_individual_hc)


        all_best_combinations_hc.append({
            'bs_index': l,
            'uav_index': k,
            'best_fitness': best_fitness_hc_current, # Store best fitness found over iterations
            'best_individual': best_individual_hc,
            'generation_fitness': best_generation_data_hc # Store fitness history for plotting
        })
        # print(f"Best Fitness for BS {l}, UAV {k} (Hill Climb): {best_fitness_hc_current:.4f}")


    # Find best individual for current BS across all UAVs
    best_individual_for_bs_hc = min(all_best_individuals_bs_hc, key=lambda x: x['fitness'])
    # print(f"Best Fitness for BS {l} across all UAVs (Hill Climb): {best_individual_for_bs_hc['fitness']:.4f}")


    # Auction-based assignment (same as before, just using HC results)
    combination_lookup_hc = {}
    for combination in all_best_combinations_hc:
        if combination['bs_index'] not in combination_lookup_hc:
            combination_lookup_hc[combination['bs_index']] = {}
        combination_lookup_hc[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments_hc = []
    unassigned_bs_hc = list(range(num_bs))
    unassigned_uavs_hc = list(range(num_uav_irs))

    while unassigned_bs_hc and unassigned_uavs_hc:
        best_combination_overall_hc = None

        for l in unassigned_bs_hc:
            best_fitness_for_bs_hc = float('inf')
            best_combination_for_bs_hc = None
            for k in unassigned_uavs_hc:
                if l in combination_lookup_hc and k in combination_lookup_hc[l]:
                    combination = combination_lookup_hc[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs_hc: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs_hc = combination['best_fitness']
                        best_combination_for_bs_hc = combination

            if best_combination_for_bs_hc:
                if best_combination_overall_hc is None or best_combination_for_bs_hc['best_fitness'] < best_combination_overall_hc['best_fitness']: # Compare with current best overall
                    best_combination_overall_hc = best_combination_for_bs_hc

        if best_combination_overall_hc:
            best_assignments_hc.append(best_combination_overall_hc)
            unassigned_bs_hc.remove(best_combination_overall_hc['bs_index'])
            unassigned_uavs_hc.remove(best_combination_overall_hc['uav_index'])

    total_fitness_for_pmax = sum(assignment['best_fitness'] for assignment in best_assignments_hc)
    fitness_sum_vs_pmax.append(total_fitness_for_pmax)
    print(f"Total Fitness for p_max = {p_max}: {total_fitness_for_pmax:.4f}")


# Plotting graphs for p_max vs Fitness Sum
plt.figure(figsize=(10, 6))
plt.plot(list(p_max_values), fitness_sum_vs_pmax, marker='o', linestyle='-')
plt.xlabel('p_max Value')
plt.ylabel('Sum of Best Fitness Values')
plt.title('Sum of Best Fitness vs. p_max (Hill Climbing)')
plt.grid(True)
plt.xticks(list(p_max_values))
plt.tight_layout()
plt.show()