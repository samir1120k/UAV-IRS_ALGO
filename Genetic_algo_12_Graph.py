#update the previous algorithm
import pandas as pd
import numpy as np
import random
import math
from numpy import random
import cmath
import matplotlib.pyplot as plt # Import matplotlib

# Load datasets related to Base Stations, UAVs, and Clients
base = pd.read_csv(r'BS_data.csv')  # Dataset containing values related to Base Stations
uav = pd.read_csv(r'UAV_data.csv')  # Dataset containing values related to UAV-IRS
people = pd.read_csv(r'people_data.csv')  # Dataset containing values related to Clients
IRS=pd.read_csv(r'IRS_data.csv') #dataset containing value of downlink transmission between base station and uav
IRS_UP=pd.read_csv(r'IRS_data_up.csv') #dataset containing value of uplink transmission of single client

# Constants
Wl = [4, 5, 6, 7, 8, 9, 10, 6]  # Weight of UAV-IRS
H = base['H']  # Height of UAV-IRS above the ground
P_m_har = base['P_m_har']  # Harvesting power of each Base Station
T_m_har = base['T_m_har']  # Harvesting time of each Base Station
P_m_down = base['P_m_down']  # Downlink power of Base Station
# T_ml_down = base['T_ml_down']  # Downlink time of Base Station to UAV-IRS
# T_km_com = people['T_km_com']  # Computation time of each client
f_km=np.random.uniform(0,10, 50) #Frequency of local computing
# T_km_up = people['T_km_up']  # Uplink time of each client
V_lm_vfly = uav['V_lm_vfly']  # Velocity of UAV-IRS during vertical flight
V_lm_hfly = uav['V_lm_hfly']  # Velocity of UAV-IRS during horizontal flight
D_l_hfly = uav['D_l_hfly']  # Horizontal distance between UAV-IRS and Base Station
Angle=IRS['Angle'] #downlink channel coefficient between base stationa and uav
h_l_km=IRS['h_l_km'] #downlink channel gain of base station and uav
h_l_m=IRS['h_l_m'] #downlink channel gain of base station and uav
P_km_up=IRS_UP['P_km_up'] #uplink trainsmission power of cliets
Angle1=IRS_UP['Angle'] #theta value for single client
g_l_km=IRS_UP['h_l_km'] #uplink transmission channel gain of single cliet
g_l_m=IRS_UP['h_l_m'] #uplink transmission channel gain of single cliet



# Additional constants for calculations
delta = 2  # For calculating p_l_b
Ar = 0.503  # Area of rotor disc
s = 0.05  # Speed of rotor
Nr = 4  # Number of rotors for UAV-IRS
V_tip = 120  # Rotor solidity
Cd = 0.5  # Drag coefficient
Af = 0.06  # Fuselage area
D_km = 0.5  # Mbits
Dm=0.49 #Mbits
B=10 #MHz #bandwidth
sigma_km=pow(10,-13) #varience of additive white Gaussian noise of base_station

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

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=h_ml_worst*P_m_down
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km**2) # it will return the sigal value which is minimum of all
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
  h_l_m.transpose()
  a=np.dot(h_l_m,diagonal)
  b=np.dot(a,h_l_km)
  final=abs(b)
  return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m**2))
    return B*math.log2(1+temp1)


#_______________________________________________________________________________________________
# Genetic Algorithm Parameters
num_bs = 10
num_iterations = 10  # Number of iterations for each generation
num_generation = 30  # Number of generations (increased to 100)
num_uav_irs = 8
all_best_combinations = []  # Store the best combination of BS and UAV
all_best_individuals = [] # Store all best individuals across all BS and UAV combinations

# Main Genetic Algorithm Loop
for l in range(num_bs):
    all_best_individuals_bs = [] # Store best individuals for current BS
    P_m_har_value = P_m_har.values[l]
    T_m_har_value = T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    # T_ml_down_value = T_ml_down.values[l]
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
            f_km_value = f_km[i]  # Randomly select computation time
            # T_km_up_value = T_km_up.values[i]  # Randomly select uplink time
            P_km_up_value=P_km_up.values[i]


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
            h_kml_down_value=h_kml_down(Angle,h_l_m,h_l_km)
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=Dm/R_ml_down_value
            # print(T_ml_down_value)
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            # print(T_km_com_value)
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
                'P_km_up_value':P_km_up_value,
                'h_kml_down_value':h_kml_down_value,
                'T_km_com_value':T_km_com_value
            }
                                    # Store parent data in the population list
            population.append({
                'fitness': result_fitness,
                'data': current_data
            })

        generations_data = [] # List to store fitness for each generation
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
                        'P_km_up_value':P_km_up_value,
                        'h_kml_down_value':h_kml_down_value,
                        'T_km_com_value':T_km_com_value
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
            generations_data.append(population[0].copy()) # Store best fitness for each generation


        best_individual_pair = population[0].copy() # Store a COPY
        best_individual_pair['generation'] = j + 1 #generation will be last j value which is incorrect
        best_individual_pair['type'] = 'GA'
        best_individual_pair['bs_index'] = l # Store BS index
        best_individual_pair['uav_index'] = k # Store UAV index
        all_best_individuals_bs.append(best_individual_pair) # Store best individual for current BS and UAV pair

        # Store generation data for current BS-UAV pair
        all_best_combinations.append({
            'bs_index': l,
            'uav_index': k,
            'best_fitness': population[0]['fitness'], # Store best fitness for this BS-UAV pair
            'best_individual': best_individual_pair,
            'generation_fitness': [gen['fitness'] for gen in generations_data] # Store fitness for each generation
        })
        # print(f"Best Fitness for BS {l}, UAV {k}: {population[0]['fitness']:.4f}") # Print fitness after each pair generation

    # Find best individual for current BS across all UAVs
    best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
    # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")

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
best_pair_for_plot = None # Variable to store the best pair for plotting
min_fitness_for_plot = float('inf')
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

    # Determine the best pair for plotting (the one with the minimum fitness among assignments)
    if assignment['best_individual']['fitness'] < min_fitness_for_plot:
        min_fitness_for_plot = assignment['best_individual']['fitness']
        best_pair_for_plot = assignment

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

# Plotting graph for the best BS-UAV pair
best_pair_combination = min(all_best_combinations, key=lambda x: x['best_fitness'])
best_bs_index_plot = best_pair_combination['bs_index']
best_uav_index_plot = best_pair_combination['uav_index']

generations = []
fitness_values = []


# Calculate sum of best fitness values for each generation across all BS-UAV pairs
sum_fitness_per_generation = [0] * num_generation
for gen_idx in range(num_generation):
    generation_sum = 0
    for combination in all_best_combinations:
        generation_sum += combination['generation_fitness'][gen_idx]
    sum_fitness_per_generation[gen_idx] = generation_sum

generation_indices = list(range(1, num_generation + 1)) # Generation numbers for x-axis

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(generation_indices, sum_fitness_per_generation, marker='o', linestyle='-')
plt.title('Sum of Best Fitness Values Across Generations')
plt.xlabel('Generation Number')
plt.ylabel('Sum of Best Fitness Values')
plt.grid(True)
plt.xticks(generation_indices) # Ensure x-axis ticks are at each generation
plt.show()

#plot the graph of single  best pair  uav  base station of best finess value across the generation 
# Plotting the graph for the best pair
if best_pair_for_plot:
    fitness_history_best_pair = best_pair_for_plot['fitness_history']
    generations = range(1, num_generation + 1) # Generation numbers for x-axis

    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_history_best_pair, marker='o', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title(f'Fitness Improvement Over Generations for Best BS-UAV Pair (BS {best_pair_for_plot['bs_index']}, UAV {best_pair_for_plot['uav_index']})')
    plt.grid(True)
    plt.xticks(generations) # Ensure all generation numbers are shown
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
else:
    print("\nNo best pair found for plotting.")