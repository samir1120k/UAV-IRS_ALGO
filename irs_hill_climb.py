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
p_km_UP=pd.read_csv(r'P_km_up.csv')
f_km1=pd.read_csv(r'f_km.csv')

# Constants
Wl = 35.28
H = 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
f_km=f_km1['0']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly = 100
P_km_up=p_km_UP['0']
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
B=10 #MHz
sigma_km=10**(-13)
eta=10
kappa=0.5
Dm=0.49 # keeping Dm global as used in the code

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
    temp1= h_ml_worst*P_m_down # Corrected: removed np.min
    if (1+temp1) <= 0:
        return 0  # Return 0 if log argument is non-positive to avoid error
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
        # the value for each itaration

def calculate_exp_i_theta(theta): # part of equation 8
    return cmath.exp(1j * theta)
 # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km, num_irs_ele): # Modified: Added num_irs_ele argument
    result=[]
    if isinstance(Angle, float):
        return 0

    if not isinstance(Angle, pd.Series):
        raise TypeError(f"Expected Angle to be pd.Series, got {type(Angle)}")

    # Slice Series to num_irs_ele inside the function
    Angle_sliced = Angle.iloc[:num_irs_ele]
    h_l_m_sliced = h_l_m.iloc[:num_irs_ele]
    h_l_km_sliced = h_l_km.iloc[:num_irs_ele]

    for i in range(len(Angle_sliced)): # Iterate over the sliced Angle
        theta_radians = math.radians(Angle_sliced.iloc[i])
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    h_l_m_np = h_l_m_sliced.to_numpy() # Convert sliced Series to numpy array
    h_l_km_np = h_l_km_sliced.to_numpy() # Convert sliced Series to numpy array
    if h_l_m_np.ndim == 1:
        h_l_m_np = h_l_m_np.reshape(1, -1)
    if h_l_km_np.ndim == 1:
        h_l_km_np = h_l_km_np.reshape(-1, 1)


    a=np.dot(h_l_m_np,diagonal)
    b=np.dot(a,h_l_km_np)
    final=abs(b[0][0])
    return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    return B*math.log2(1+temp1)

def sub(P_i_up,h_il_up):
    return P_i_up*h_il_up

def E_km_com(f_km,T_km_com):
    return eta*(10**(-28))*(f_km**3)*T_km_com

def E_kml_up(P_km_up,T_km_up):
    return P_km_up*T_km_up

def E_kml_har(P_m_har, T_m_har): 
    return kappa*P_m_har*T_m_har

num_population=50
num_bs = 5
num_generation = 1
num_uav_irs = 8
population_size = 50

numerical_keys_for_crossover = [
    'P_m_down_value', 'P_m_har_value', 'T_m_har_value',
    'f_km_value', 'V_lm_vfly_value', 'V_lm_hfly_value',
    'P_km_up_value','f_km_value',
]

fitness_sums_irs_ele = [] # Store sum of fitness values for each num_irs_ele
irs_element_values = range(10, 51, 5) # num_irs_ele values from 10 to 50 in steps of 5


for num_irs_ele in irs_element_values: # Loop through different num_irs_ele values
    print(f"Calculation for num_irs_ele: {num_irs_ele}")

    Angle_df=pd.read_csv(r'Angle.csv') # number of IRS is 500 store in each column
    h_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 500 store in each column
    h_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 500 store in each column

    Angle_UP_df=pd.read_csv(r'Angle1.csv') # number of IRS is 500 store in each column
    g_l_km_df=pd.read_csv(r'h_l_km1.csv') # number of IRS is 500 store in each column
    g_l_m_df=pd.read_csv(r'h_l_m1.csv') # number of IRS is 500 store in each column # corrected filename

    Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 500 store in each column
    f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 500 store in each column
    f_l_m_df=pd.read_csv(r'h_l_m2.csv') # number of IRS is 500 store in each column # corrected filename


    all_best_combinations = []
    all_best_individuals = []

    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]
        H_value = H

        index_list = list(range(500))
        random.shuffle(index_list)
        unique_row_indices = index_list[:population_size]

        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :num_irs_ele].reset_index(drop=True) # Sliced columns
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :num_irs_ele].reset_index(drop=True) # Sliced columns
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :num_irs_ele].reset_index(drop=True) # Sliced columns
        f_km_bs = f_km[unique_row_indices].reset_index(drop=True)
        valid_indices = [i for i in unique_row_indices if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            D_l_hfly_value = D_l_hfly
            Wl_value = Wl
            Sub_value=0

            for i in range(len(valid_indices)):
                # Slice the Angle_UP_df, g_l_m_df, g_l_km_df_bs to num_irs_ele columns
                Angle_UP_df_sliced = Angle_UP_df.iloc[i, :num_irs_ele]
                g_l_m_df_sliced = g_l_m_df.iloc[k, :num_irs_ele]
                g_l_km_df_bs_sliced = g_l_km_df_bs.iloc[i, :num_irs_ele]

                h_il_up_value=h_kml_down(Angle_UP_df_sliced, g_l_m_df_sliced, g_l_km_df_bs_sliced, num_irs_ele) # Pass num_irs_ele
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)


            for i in range(len(valid_indices)):
                f_km_value = f_km_bs[i]
                P_km_up_value = P_km_up_bs[i]

                # Slice the Angle_df, h_l_m_df, h_l_km_df_bs, Angle_har_df, f_l_m_df, f_l_km_df_bs to num_irs_ele columns
                Angle_row = Angle_df.iloc[i, :num_irs_ele]
                h_l_m_row = h_l_m_df.iloc[k, :num_irs_ele]
                h_l_km_row = h_l_km_df_bs.iloc[i, :num_irs_ele]
                Angle1_row = Angle_UP_df.iloc[i, :num_irs_ele] # using sliced version not needed again
                g_l_m_row = g_l_m_df.iloc[k, :num_irs_ele]  # using sliced version not needed again
                g_l_km_row = g_l_km_df_bs.iloc[i, :num_irs_ele] # using sliced version not needed again
                Angle2_row = Angle_har_df.iloc[i, :num_irs_ele]
                f_l_m_row = f_l_m_df.iloc[k, :num_irs_ele]
                f_l_km_row = f_l_km_df_bs.iloc[i, :num_irs_ele]

                Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                Bh = max(1, Bh)
                p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row, num_irs_ele) # Pass num_irs_ele
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=Dm/R_ml_down_value # using global Dm
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row, num_irs_ele) # Pass num_irs_ele

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=Dm/R_kml_up_value # using global Dm
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'f_km_value': f_km_value,
                        'T_km_up_value': T_km_up_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'Angle1_row':Angle1_row,
                        'Angle_row':Angle_row,
                        'Angle2_row': Angle2_row,
                        }
                    })

            generations_data = []
            for j in range(num_generation):
                child_population = []
                                    # Corrected loop range to use valid_indices length
                for i in range(0, len(valid_indices), 2): # Loop through population with step of 2
                    if i + 1 >= len(valid_indices): # Check if i+1 is within bounds, if not break to avoid error in accessing population[i+1]
                        break
                    ranodmpopulation=[]
                    for i in range(10):
                        ranodmpopulation.append(random.choice(population))
                    ranodmpopulation = sorted(ranodmpopulation, key=lambda x: x['fitness'])
                    parent1 = ranodmpopulation[0]
                    parent2 = ranodmpopulation[1]
                    child_data = {}

 
                    for key in numerical_keys_for_crossover: # Apply mutation only to numerical keys
                        child_data[key] += random.normal(loc=0, scale=1, size=(1))[0]


                def compute_fitness(data):
                    P_m_down_value = data['P_m_down_value']
                    P_m_har_value = data['P_m_har_value']
                    T_m_har_value = data['T_m_har_value']
                    f_km_value = data['f_km_value']
                    T_km_up_value = data['T_km_up_value']
                    V_lm_vfly_value = data['V_lm_vfly_value']
                    V_lm_hfly_value = data['V_lm_hfly_value']
                    P_km_up_value=data['P_km_up_value']
                    Angle_row = data['Angle_row']
                    Angle1_row = data['Angle1_row']
                    Angle2_row = data['Angle2_row']


                    Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                    Bh = max(1, Bh)
                    p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                    P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                    P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                    P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                    T_l_vfly_value = H_value / V_lm_vfly_value
                    T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
                    E_ml_har_value = P_m_har_value * T_m_har_value

                    h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row, num_irs_ele) # Pass num_irs_ele
                    h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                    R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                    if R_ml_down_value <= 0:
                        R_ml_down_value = 1e-9
                    T_ml_down_value=Dm/R_ml_down_value # using global Dm
                    E_ml_down_value = P_m_down_value * T_ml_down_value
                    T_km_com_value = D_km / f_km_value
                    T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                    P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                    P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                    h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row, num_irs_ele) # Pass num_irs_ele
                    E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value) # Corrected: removed h_kml_har
                    E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                    E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)

                    fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                    current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'T_km_up_value': T_km_up_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value':P_km_up_value,
                            'Angle1_row':Angle1_row,
                            'Angle_row':Angle_row,
                            'Angle2_row': Angle2_row,
                                                }
                    if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                        return fitness_value, current_data
                    else:
                        return  float('inf'),{}

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
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
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
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    sum_fitness_current_p_max = 0 # Sum of best fitness for current p_max
    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value}") # Print the entire Series directly - corrected line
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values

        print("-" * 20)
        sum_fitness_current_p_max += best_ind['fitness'] # Sum fitness values

        if assignment['best_individual']['fitness'] < min_fitness_for_plot:
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    fitness_sums_irs_ele.append(sum_fitness_current_p_max) # Store sum of fitness for this p_max

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

# Plotting graph for Sum of Fitness vs num_irs_ele
plt.figure(figsize=(10, 6))
plt.plot(irs_element_values, fitness_sums_irs_ele, marker='o', linestyle='-')
plt.xlabel('Number of IRS Elements (num_irs_ele)')
plt.ylabel('Sum of Best Fitness Values (Auctioned Assignments)')
plt.title('Sum of Best Fitness Values vs Number of IRS Elements')
plt.grid(True)
plt.xticks(irs_element_values)
plt.tight_layout()
plt.show()