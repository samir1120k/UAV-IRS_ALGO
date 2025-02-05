import pandas as pd
import numpy as np
import random 
import math
from numpy import random

base=pd.read_csv(r'BS_data.csv')
uav=pd.read_csv(r'UAV_data.csv')
people=pd.read_csv(r'people_data.csv')


Wl=[4,5,6,7,8,9,10,6]
H=base['H']
P_m_har=base['P_m_har']
T_m_har=base['T_m_har']
P_m_down=base['P_m_down']
T_ml_down=base['T_ml_down']
T_km_com=people['T_km_com']
T_km_up=people['T_km_up']
V_lm_vfly=uav['V_lm_vfly']
V_lm_hfly=uav['V_lm_hfly']
D_l_hfly=uav['D_l_hfly']

delta=2 #for calculat the p_l_b  for now
Ar=0.503 #area of rotor disc
s= 0.05 #Speed of rotor
Nr=4 #number of rotor fo UAV-IRS
V_tip= 120 #rotor solidity 120
Cd=0.5 #take a loop 10 value bwtween .02 to 1
Af=0.06 #fuselage area in equation 15 0.06
D_km=0.5 #Mbits


def Fitness(E_ml_har,E_ml_down,E_ml_UAV): #eqation number 30
    return E_ml_har+E_ml_down+E_ml_UAV # it should be m*l dimension

def E_ml_UAV(P_l_vfly,T_l_vfly,P_lm_hfly,T_l_hfly,P_l_hov,T_lm_hov): #energy consumption of the UAV-IRS,
    return P_l_vfly*T_l_vfly+P_lm_hfly*T_l_hfly+P_l_hov*T_lm_hov # eqation number 20

def P_l_vfly(Wl,V_l_vfly,P_l_b,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=np.sqrt(V_l_vfly**2+(2*Wl)/temp2)
    return ((Wl/2)*(V_l_vfly+temp3))+Nr*P_l_b

def P_lm_hfly(P_lm_blade,P_lm_fuselage,P_lm_induced): #eqation number 13
    return P_lm_blade+P_lm_fuselage+P_lm_induced

def P_lm_blade(Nr,P_l_b,V_tip,V_lm_hfly): #eqation number 14
    return Nr*P_l_b*(1+((3*(V_lm_hfly**2))/pow(V_tip,2)))

def P_lm_fuselage(Cd,Af,Bh,V_lm_hfly): #eqation number 15
    return (1/2)*Cd*Af*Bh*(V_lm_hfly**3)

def P_lm_induced(Nr,Bh,Ar,Wl,V_lm_hfly): #eqation number 16
    return Wl*(np.sqrt((Wl**2)/(4*(Nr**2)*(Bh**2)*(Ar**2))+((V_lm_hfly**4)/4))-((V_lm_hfly**2)/2)**(1/2))

def P_l_hov(Wl,P_l_b,Nr,Ar,Bh): #eqation number 18
    temp1=Nr*P_l_b
    temp2=abs(2*(Nr*Bh*Ar))
    temp3=np.sqrt(temp2)
    temp4=(Wl**3/2)/temp3
    return temp1+ temp4

def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down   #we have to take the max of T_com_km and max 

num_bs=10
num_iterations = 10  # Or however many iterations you want
num_generation=10
num_uav_irs=8
all_best_combinations = []  # Store the best combination of BS and UAV

for l in range(num_bs):
    all_best_individuals=[]
    P_m_har_value = P_m_har.values[l]
    T_m_har_value =T_m_har.values[l]
    P_m_down_value = P_m_down.values[l]
    T_ml_down_value =T_ml_down.values[l]
    H_value =H.values[l]
    for k in range(num_uav_irs):
        best_fitness = float('inf')  # Initialize with infinity
        best_individual = {}  # Store the individual with the best fitness
        population=[]
        V_lm_vfly_value = V_lm_vfly.values[k]
        V_lm_hfly_value = V_lm_hfly.values[k]
        D_l_hfly_value = D_l_hfly.values[k]
        Wl_value = Wl[k]

        for j in range(num_generation):
            parent1_fitness = None  # Store fitness value
            parent1_data = {}      # Store corresponding variables
            parent2_fitness = None  # Store fitness value
            parent2_data = {}      # Store corresponding variables
            child_fitness=None
            child_data1={}


            for i in range(num_iterations):
                f_km_value = random.choice(T_km_com.values)
                T_km_up_value = random.choice(T_km_up.values)

                Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                Bh = max(1, Bh)
                p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value * V_lm_hfly_value
                E_ml_har_value = P_m_har_value * T_m_har_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value=D_km/f_km_value
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)


                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)  # Calculate fitness

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

                if parent1_fitness is None:
                    parent1_fitness = result_fitness
                    parent1_data = current_data
                elif parent2_fitness is None:
                    parent2_fitness = result_fitness
                    parent2_data = current_data
                else:  # Update logic (example: keep the two best)
                    if result_fitness < parent1_fitness:
                        parent2_fitness = parent1_fitness
                        parent2_data = parent1_data
                        parent1_fitness = result_fitness
                        parent1_data = current_data
                    elif result_fitness < parent2_fitness:
                        parent2_fitness = result_fitness
                        parent2_data = current_data

            # print("Parent 1:")
            # print(f"  Fitness: {parent1_fitness}")
            # for key, value in parent1_data.items():
            #     print(f"  {key}: {value}")

            # print("\nParent 2:")
            # print(f"  Fitness: {parent2_fitness}")


            # for key, value in parent2_data.items():
            #     print(f"  {key}: {value}")

                # Store parent data in the population list
            population.append({
                'generation': j + 1,  # Add generation number
                'type': 'parent1',
                'fitness': parent1_fitness,
                'data': parent1_data
            })
            population.append({
                'generation': j + 1,
                'type': 'parent2',
                'fitness': parent2_fitness,
                'data': parent2_data
            })

            
            # Crossover Process 
            child_data = {
                key: parent1_data[key] * 0.6 + parent2_data[key] * (1 - 0.6)
                for key in parent1_data
            }
            #mutaion process

            #lets suppose for now Mutation is between 0 to 1
            # randomly choose the key value pair for mutation
            # keys = list(child_data.keys())
            # random_key = random.choice(keys)
            # child_data[random_key] = child_data[random_key] + np.random.uniform(-1, 1, 1)[0] # [0] to get the single value

            # keys1 = list(child_data.keys())
            # random_key1 = random.choice(keys1)
            # child_data[random_key1] = child_data[random_key1] + np.random.uniform(-1, 1, 1)[0] # [0] to get the single value

            # Method 3: Update all key-value pairs at once (more efficient)
            u=np.random.uniform(0, 1, 1)[0]
            P_mutation=0.5
            if u< P_mutation:
                for key in child_data:
                    child_data[key] = child_data[key] + random.normal(loc=0, scale=1, size=(1))

            # print("Updated child_data (Method 3):", child_data)

            # compute the fitness of child and store in the child_fitness


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

                Bh = (1 - 2.2558 * pow(10, 4) * H_value)
                Bh = max(1, Bh)
                p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_hfly_value)

                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value * V_lm_hfly_value
                E_ml_har_value = P_m_har_value * T_m_har_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value=D_km/f_km_value
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)



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
                return fitness_value,current_data


                
            child_fitness, child_data1=compute_fitness(child_data) #compute child fitness
            # print("Child Fitness:", child_fitness)
            # print("child Data : ",child_data1)

            # Store child data in the population list
            population.append({
                'generation': j + 1,
                'type': 'child',
                'fitness': child_fitness,
                'data': child_data1
            })

            # Check for best fitness and update
            for individual in population[-3:]: # Check the last 3 added (parents and child)
                if individual['fitness'] < best_fitness:
                    best_fitness = individual['fitness']
                    best_individual = individual.copy()  # Important: Create a COPY
            

        # Store the best individual for this BS-UAV combination



        # i have to store the parant1 and paranet 2 and child detail  and create the population

        # After the loop, print the entire population data
        # print("\n--- Population Data ---")
        # for individual in population:
        #     print(f"Generation: {individual['generation']}, Type: {individual['type']}")
        #     print(f"  Fitness: {individual['fitness']}")
        #     for key, value in individual['data'].items():
        #         print(f"  {key}: {value}")
        #     print("-" * 20)  # Separator between individuals

        #select the best fitness among them all 
        # Print the best individual found
        # print("\n--- Best Individual ---")
        # print(f"Generation: {best_individual['generation']}, Type: {best_individual['type']}")
        # print(f"  Fitness: {best_individual['fitness']}")
        # for key, value in best_individual['data'].items():
        #     print(f"  {key}: {value}")
        
        all_best_individuals.append(best_individual.copy())  # Store a COPY

        all_best_combinations.append({
    'bs_index': l,
    'uav_index': k,
    'best_individual': best_individual.copy()
    })


    # Print all best individuals found across all rounds
    # print("\n--- Best Individuals Across All Rounds ---")
    # for i, best_ind in enumerate(all_best_individuals):
    #     print(f"\nBest Individual (Round {i+1}):")
    #     print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
    #     print(f"  Fitness: {best_ind['fitness']}")
    #     for key, value in best_ind['data'].items():
    #         print(f"  {key}: {value}")
    #     print("-" * 20)

    # If you want to find the *overall* best individual across all rounds:
    # overall_best_fitness = float('inf')
    # overall_best_individual = {}

    # for individual in all_best_individuals:
    #     if individual['fitness'] < overall_best_fitness:
    #         overall_best_fitness = individual['fitness']
    #         overall_best_individual = individual.copy()

    # print("\n--- Overall Best Individual ---")
    # print(f"Generation: {overall_best_individual['generation']}, Type: {overall_best_individual['type']}")
    # print(f"  Fitness: {overall_best_individual['fitness']}")
    # for key, value in overall_best_individual['data'].items():
    #     print(f"  {key}: {value}")
    # print("-" * 20)



# Find the overall best combination across all BS and UAV pairs
# overall_best_fitness = float('inf')
# overall_best_combination = {}

# for combination in all_best_combinations:
#     if combination['best_individual']['fitness'] < overall_best_fitness:
#         overall_best_fitness = combination['best_individual']['fitness']
#         overall_best_combination = combination.copy()

# Print the overall best combination
# print("\n--- Overall Best Combination (BS and UAV) ---")
# print(f"BS Index: {overall_best_combination['bs_index']}")
# print(f"UAV Index: {overall_best_combination['uav_index']}")
# best_ind = overall_best_combination['best_individual']
# print(f"Best Individual:")
# print(f" Generation: {best_ind['generation']}, Type: {best_ind['type']}")
# print(f" Fitness: {best_ind['fitness']}")
# for key, value in best_ind['data'].items():
#     print(f" {key}: {value}")
# print("-" * 20)



# Print all best combinations (optional)
# print("\n--- Best Combinations for each BS-UAV Pair ---")  # Uncomment if you want to see all
# for i, combination in enumerate(all_best_combinations):
#     print(f"\nBest Combination (BS {combination['bs_index']}, UAV {combination['uav_index']}):")
#     best_ind = combination['best_individual']
#     print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
#     print(f"  Fitness: {best_ind['fitness']}")
#     for key, value in best_ind['data'].items():
#         print(f"  {key}: {value}")
#     print("-" * 20)

# Assign best UAV to each BS (ensuring unique assignments)

# Optimization: Create a dictionary to quickly lookup combinations by BS index and UAV index
combination_lookup = {}
for combination in all_best_combinations:
    if combination['bs_index'] not in combination_lookup:
        combination_lookup[combination['bs_index']] = {}
    combination_lookup[combination['bs_index']][combination['uav_index']] = combination

# Assign best UAV to each BS (ensuring unique assignments)

best_assignments = []
unassigned_bs = list(range(num_bs))  # List of unassigned BS indices
unassigned_uavs = list(range(num_uav_irs)) # List of unassigned UAV indices


while unassigned_bs and unassigned_uavs:
    best_combination_overall = None

    for l in unassigned_bs:
        best_fitness_for_bs = float('inf')
        best_combination_for_bs = None
        for k in unassigned_uavs:
            # Use the lookup dictionary for O(1) access instead of iterating through all combinations
            if l in combination_lookup and k in combination_lookup[l]: # check if bs and uav are in the lookup
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
    print(f"  UAV Index: {assignment['uav_index']}")
    best_ind = assignment['best_individual']
    print(f"  Best Individual:")
    print(f"    Generation: {best_ind['generation']}, Type: {best_ind['type']}")
    print(f"    Fitness: {best_ind['fitness']}")
    for key, value in best_ind['data'].items():
        print(f"    {key}: {value}")
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