import pandas as pd
import random
import math

base=pd.read_csv(r'BS_data.csv')
uav=pd.read_csv(r'UAV_data.csv')
people=pd.read_csv(r'people_data.csv')


Wl=[4,5,6,7,8,9,10]
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


def Fitness(E_ml_har,E_ml_down,E_ml_UAV): #eqation number 30
    return E_ml_har+E_ml_down+E_ml_UAV # it should be m*l dimension

def E_ml_UAV(P_l_vfly,T_l_vfly,P_lm_hfly,T_l_hfly,P_l_hov,T_lm_hov): #energy consumption of the UAV-IRS,
    return P_l_vfly*T_l_vfly+P_lm_hfly*T_l_hfly+P_l_hov*T_lm_hov # eqation number 20

def P_l_vfly(Wl,V_l_vfly,P_l_b,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(pow(V_l_vfly,2)+(2*Wl)/temp2)
    return ((Wl/2)*(V_l_vfly+temp3))+Nr*P_l_b

def P_lm_hfly(P_lm_blade,P_lm_fuselage,P_lm_induced): #eqation number 13
    return P_lm_blade+P_lm_fuselage+P_lm_induced

def P_lm_blade(Nr,P_l_b,V_tip,V_lm_hfly): #eqation number 14
    return Nr*P_l_b*(1+((3*pow(V_lm_hfly,2))/pow(V_tip,2)))

def P_lm_fuselage(Cd,Af,Bh,V_lm_hfly): #eqation number 15
    return (1/2)*Cd*Af*Bh*pow(V_lm_hfly,3)

def P_lm_induced(Nr,Bh,Ar,Wl,V_lm_hfly): #eqation number 16
    return Wl*pow(math.sqrt(pow(Wl,2)/(4*pow(Nr,2)*pow(Bh,2)*pow(Ar,2))+(pow(V_lm_hfly,4)/4))-(pow(V_lm_hfly,2)/2),(1/2))

def P_l_hov(Wl,P_l_b,Nr,Ar,Bh): #eqation number 18
    temp1=Nr*P_l_b
    temp2=abs(2*(Nr*Bh*Ar))
    temp3=math.sqrt(temp2)
    temp4=pow(Wl,3/2)/temp3
    return temp1+ temp4

def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down   #we have to take the max of T_com_km and max 
population_size = 20  # Number of solutions in each generation
num_generations = 50   # Number of generations to evolve
mutation_rate = 0.1   # Probability of mutation for each parameter

def crossover(parent1_data, parent2_data):
    child_data = {}
    for key in parent1_data:
        if key in ['H_value','Wl_value','P_m_down_value','P_m_har_value','T_m_har_value','T_ml_down_value','T_km_com_value','T_km_up_value','V_lm_vfly_value','V_lm_hfly_value','D_l_hfly_value']:
            child_data[key] = min(parent1_data[key], parent2_data[key])
        else:
            child_data[key] = random.choice([parent1_data[key], parent2_data[key]])
    return child_data

def mutate(data):
    mutated_data = data.copy()
    for key in mutated_data:
        if random.random() < mutation_rate:
            if isinstance(mutated_data[key], (int, float)):
                range_val = abs(mutated_data[key] * 0.1)  # 10% range
                mutated_data[key] += random.uniform(-range_val, range_val)
            elif isinstance(mutated_data[key], pd.Series):
                index = random.randint(0, len(mutated_data[key]) - 1)
                mutated_data[key].iloc[index] = random.uniform(mutated_data[key].min(), mutated_data[key].max())
    return mutated_data

# Initialize population
population = []
for _ in range(population_size):
    individual_data = {}
    H_value = random.choice(H.values)
    Wl_value = random.choice(Wl)
    P_m_down_value = random.choice(P_m_down.values)
    P_m_har_value = random.choice(P_m_har.values)
    T_m_har_value = random.choice(T_m_har.values)
    T_ml_down_value = random.choice(T_ml_down.values)
    T_km_com_value = random.choice(T_km_com.values)
    T_km_up_value = random.choice(T_km_up.values)
    V_lm_vfly_value = random.choice(V_lm_vfly.values)
    V_lm_hfly_value = random.choice(V_lm_hfly.values)
    D_l_hfly_value = random.choice(D_l_hfly.values)

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
    T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
    P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
    P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
    P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
    E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

    individual_data = { #store all values in dictionary
        'H_value': H_value, 'Wl_value': Wl_value, 'P_m_down_value': P_m_down_value,
        'P_m_har_value': P_m_har_value, 'T_m_har_value': T_m_har_value,
        'T_ml_down_value': T_ml_down_value, 'T_km_com_value': T_km_com_value,
        'T_km_up_value': T_km_up_value, 'V_lm_vfly_value': V_lm_vfly_value,
        'V_lm_hfly_value': V_lm_hfly_value, 'D_l_hfly_value': D_l_hfly_value,
        'Bh': Bh, 'p_l_b': p_l_b, 'P_lm_blade_value': P_lm_blade_value,
        'P_lm_fuselage_value': P_lm_fuselage_value, 'P_lm_induced_value': P_lm_induced_value,
        'T_l_vfly_value': T_l_vfly_value, 'T_l_hfly_value': T_l_hfly_value,
        'E_ml_har_value': E_ml_har_value, 'E_ml_down_value': E_ml_down_value,
        'T_lm_hov_value': T_lm_hov_value, 'P_l_hov_value': P_l_hov_value,
        'P_l_vfly_value': P_l_vfly_value, 'P_lm_hfly_value': P_lm_hfly_value,
        'E_ml_UAV_value': E_ml_UAV_value
    }
    individual_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
    population.append({'data': individual_data, 'fitness': individual_fitness})

# Genetic Algorithm Loop
for generation in range(num_generations):
    # Selection (Tournament Selection)
    selected_parents = []
    for _ in range(population_size):
        tournament = random.sample(population, 5)  # Tournament size 5
        winner = max(tournament, key=lambda x: x['fitness'])
        selected_parents.append(winner['data'])

    new_population = []
    for i in range(0, population_size, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1] if i + 1 < population_size else selected_parents[0]

        child1_data = crossover(parent1, parent2)
        child2_data = crossover(parent1, parent2)

        child1_data = mutate(child1_data)
        child2_data = mutate(child2_data)

        # Evaluate child fitness (same as in initialization)
# Evaluate child 1 fitness
        Bh_child1 = (1 - 2.2558 * pow(10, 4) * child1_data['H_value'])
        Bh_child1 = max(1, Bh_child1)
        p_l_b_child1 = (delta / 8) * Bh_child1 * Ar * s * pow(V_tip, 3)
        P_lm_blade_child1 = P_lm_blade(Nr, p_l_b_child1, V_tip, child1_data['V_lm_hfly_value'])
        P_lm_fuselage_child1 = P_lm_fuselage(Cd, Af, Bh_child1, child1_data['V_lm_hfly_value'])
        P_lm_induced_child1 = P_lm_induced(Nr, Bh_child1, Ar, child1_data['Wl_value'], child1_data['V_lm_hfly_value'])

        T_l_vfly_child1 = child1_data['H_value'] / child1_data['V_lm_vfly_value']
        T_l_hfly_child1 = child1_data['D_l_hfly_value'] * child1_data['V_lm_hfly_value']
        E_ml_har_child1 = child1_data['P_m_har_value'] * child1_data['T_m_har_value']
        E_ml_down_child1 = child1_data['P_m_down_value'] * child1_data['T_ml_down_value']
        T_lm_hov_child1 = T_lm_hov(child1_data['T_km_com_value'], child1_data['T_km_up_value'], child1_data['T_ml_down_value'])
        P_l_hov_child1 = P_l_hov(child1_data['Wl_value'], p_l_b_child1, Nr, Ar, Bh_child1)
        P_l_vfly_child1 = P_l_vfly(child1_data['Wl_value'], child1_data['V_lm_vfly_value'], p_l_b_child1, Nr, Ar, Bh_child1)
        P_lm_hfly_child1 = P_lm_hfly(P_lm_blade_child1, P_lm_fuselage_child1, P_lm_induced_child1)
        E_ml_UAV_child1 = E_ml_UAV(P_l_vfly_child1, T_l_vfly_child1, P_lm_hfly_child1, T_l_hfly_child1, P_l_hov_child1, T_lm_hov_child1)

        child1_fitness = Fitness(E_ml_har_child1, E_ml_down_child1, E_ml_UAV_child1)


        # Evaluate child 2 fitness
        Bh_child2 = (1 - 2.2558 * pow(10, 4) * child2_data['H_value'])
        Bh_child2 = max(1, Bh_child2)
        p_l_b_child2 = (delta / 8) * Bh_child2 * Ar * s * pow(V_tip, 3)
        P_lm_blade_child2 = P_lm_blade(Nr, p_l_b_child2, V_tip, child2_data['V_lm_hfly_value'])
        P_lm_fuselage_child2 = P_lm_fuselage(Cd, Af, Bh_child2, child2_data['V_lm_hfly_value'])
        P_lm_induced_child2 = P_lm_induced(Nr, Bh_child2, Ar, child2_data['Wl_value'], child2_data['V_lm_hfly_value'])

        T_l_vfly_child2 = child2_data['H_value'] / child2_data['V_lm_vfly_value']
        T_l_hfly_child2 = child2_data['D_l_hfly_value'] * child2_data['V_lm_hfly_value']
        E_ml_har_child2 = child2_data['P_m_har_value'] * child2_data['T_m_har_value']
        E_ml_down_child2 = child2_data['P_m_down_value'] * child2_data['T_ml_down_value']
        T_lm_hov_child2 = T_lm_hov(child2_data['T_km_com_value'], child2_data['T_km_up_value'], child2_data['T_ml_down_value'])
        P_l_hov_child2 = P_l_hov(child2_data['Wl_value'], p_l_b_child2, Nr, Ar, Bh_child2)
        P_l_vfly_child2 = P_l_vfly(child2_data['Wl_value'], child2_data['V_lm_vfly_value'], p_l_b_child2, Nr, Ar, Bh_child2)
        P_lm_hfly_child2 = P_lm_hfly(P_lm_blade_child2, P_lm_fuselage_child2, P_lm_induced_child2)
        E_ml_UAV_child2 = E_ml_UAV(P_l_vfly_child2, T_l_vfly_child2, P_lm_hfly_child2, T_l_hfly_child2, P_l_hov_child2, T_lm_hov_child2)

        child2_fitness = Fitness(E_ml_har_child2, E_ml_down_child2, E_ml_UAV_child2)


        new_population.append({'data': child1_data, 'fitness': child1_fitness})
        new_population.append({'data': child2_data, 'fitness': child2_fitness})

    population = new_population  # Replace old population with the new one

# After all generations, find the best solution:
best_solution = min(population, key=lambda x: x['fitness'])

print("Best Solution:")
print(f"  Fitness: {best_solution['fitness']}")
for key, value in best_solution['data'].items():
    print(f"  {key}: {value}")