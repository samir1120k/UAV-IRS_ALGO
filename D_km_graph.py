# coding: utf-8
#update the previous algorithm
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

Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 50 store in each column
f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 50 store in each column
f_l_m_df=pd.read_csv(r'h_l_m2.csv') # number of IRS is 50 store in each column


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
# Angle=IRS['Angle'] # Not needed as we load from CSV now
# h_l_km=IRS['h_l_km'] # Not needed as we load from CSV now
# h_l_m=IRS['h_l_m'] # Not needed as we load from CSV now
P_km_up=IRS_UP['P_km_up']
Angle1_col=IRS_UP['Angle'] # Renamed to avoid confusion with Angle dataframe
g_l_km_col=IRS_UP['h_l_km'] # Renamed
g_l_m_col=IRS_UP['h_l_m'] # Renamed
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
D_km=0.5
B=10 #MHz
sigma_km=10**(-13)
eta=10
kappa=0.5
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
    b=np.dot(a,h_l_km_np)     # Use numpy arrays for dot product
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
num_generation = 50 # Number of generations
num_uav_irs = 8
population_size = 50 # Population size for GA

D_m_values = np.arange(0.1, 1.1, 0.1) # D_km values from 0.1 to 1
D_km_fitness_sums_total = [] # To store sum of fitness for each D_km


for D_m_current in D_m_values: # Iterate over D_km values
    all_best_combinations = []
    all_best_individuals = []
    best_fitness_overall_D_km = float('inf') # Initialize best fitness for current D_km

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

                Angle_row = Angle_df.iloc[i, :] # Get row as Series
                h_l_m_row = h_l_m_df.iloc[i, :] # Get row as Series
                h_l_km_row = h_l_km_df.iloc[i, :] # Get row as Series
                Angle1_row = Angle_UP_df.iloc[i, :] # Get row as Series
                g_l_m_row = g_l_m_df.iloc[i, :] # Get row as Series
                g_l_km_row = g_l_km_df.iloc[i, :] # Get row as Series


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
                T_ml_down_value=D_m_current/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value # Use current D_km value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=D_m_current/R_kml_up_value # equation number 5
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
                        Angle_row_compute = Angle_df.iloc[i, :]  # Use same angle row as parent - or you can introduce angle variation here if needed in GA
                        h_l_m_row_compute = h_l_m_df.iloc[i, :]
                        h_l_km_row_compute = h_l_km_df.iloc[i, :]

                        Angle1_row_compute = Angle_har_df.iloc[i, :]  # Use same angle row as parent - or you can introduce angle variation here if needed in GA
                        f_l_m_row_compute = f_l_m_df.iloc[i, :]
                        f_l_km_row_compute = f_l_km_df.iloc[i, :]

                        h_kml_down_value_compute=h_kml_down(Angle_row_compute,h_l_m_row_compute,h_l_km_row_compute) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=D_m_current/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km/ f_km_value # Use current D_km value
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
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and E_kml_har_value>=(E_kml_up_value+E_kml_com_value):
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


            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            # Store best fitness for each BS-UAV combination at this D_km
            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'], # Best fitness of the last generation
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

    # For each D_km, after auction, sum the best fitness values of all assignments
    current_D_km_fitness_sum = 0
    for assignment in best_assignments:
        current_D_km_fitness_sum += assignment['best_fitness']

    D_km_fitness_sums_total.append(current_D_km_fitness_sum) # Append the sum of fitness for plotting

    print(f"D_km: {D_m_current}, Sum of Fitness: {current_D_km_fitness_sum:.4f}")


# Plotting graph for Sum of Fitness vs D_km
plt.figure(figsize=(10, 6))
plt.plot(D_m_values, D_km_fitness_sums_total, marker='o', linestyle='-')
plt.xlabel('D_km Value (km)')
plt.ylabel('Sum of Fitness Value')
plt.title('Sum of Fitness Value vs D_km')
plt.grid(True)
plt.xticks(D_m_values)
plt.tight_layout()
plt.show()