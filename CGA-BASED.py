import numpy as np
import random
import math

#define required variable
Cd=None #drag coefficient
Af=None #fuselage area
Wl=None #weight of UAV-IRS
Nr=4 #number of rotor fo UAV-IRS 
delta=None
V_tip=None #rotor solidity
H=None #height of UAV-IRS
Ar=None #area of rotor disc
s=None #Speed of rotor
Bh=(1-2.2558*pow(10,4)*H) #air density at height H
p_b_l=(delta/8)*Bh*Ar*s*pow(V_tip,3) 
sigma_km=pow(10,-13) #varience of additive white Gaussian noise

#eqation number 18
def P_hov_lH(Wl,P_b_l,Nr,Ar,Bh):
    temp1=Nr*P_b_l
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(2*temp2)
    temp4=pow(Wl,3/2)/temp3
    return temp1+ temp4

B=10 #MHz #bandwidth
def R_down_ml(B,P_down_m,h_worst_ml): #eqation number 7
    temp1=h_worst_ml*P_down_m
    return B*math.log2(1+temp1)

def h_worst_ml(h_down_kml,sigma_km): #eqation number 8
    return min(h_down_kml/pow(pow(sigma_km),2),1) # W # for minimum selection we put the value of 1

def h_down_kml(h_l_km,V_down_lm,h_lm): #Argument will be define later
    return abs(h_l_km*V_down_lm*h_lm)

Dm=0.49 # Mbits
T_down_ml=Dm/R_down_ml #eqation number 9


def P_down_m(Dm,T_com_km_str,T_up_kml_str,Tm,h_worst_ml): #eqation number 25
    temp1=pow(((Dm/T_com_km_str*T_up_kml_str*Tm)-1),2)
    return temp1/h_worst_ml
    #condition have to be satisfied of eqation number 26
    #0<P_down_m<=P_max_m
    #still some value is not defined in eqation number 25
    #T_com_km_str
    #T_up_kml_str
    #Tm

V_vfly_l = None #m/s express as follow[35]


def P_vfly_l(Wl,V_vfly_l,P_b_l,Nr,Ar,Bh): #eqation number 11
    temp2=Nr*Bh*Ar
    temp3=math.sqrt(pow(V_vfly_l,2)+(2*Wl)/temp2)
    return ((Wl/2)*(V_vfly_l+temp3))+Nr*P_b_l

V_hfly_lm = None # forward flight to the HP is derived using teh axial momentum theory express as follow[38]

def P_blade_lm(Nr,P_B_l,V_tip,V_hfly_lm): #eqation number 14
    return Nr*P_B_l*(1+(3*pow(V_hfly_lm))/pow(V_tip,2))

def P_fuselage_lm(Cd,Af,Bh,V_hfly_lm): #eqation number 15
    return (1/2)*Cd*Af*Bh*pow(V_hfly_lm,3)

def P_induced_lm(Nr,Bh,Ar,Wl,V_hfly_lm): #eqation number 16
    return Wl*pow(math.sqrt(pow(Wl,2)/(4*pow(Nr,2)*pow(Bh,2)*pow(Ar,2))+(pow(V_hfly_lm,4)/4))-(pow(V_hfly_lm,2)/2),(1/2))


def P_hfly_lm(P_blade_lm,P_fuselage_lm,P_induced_lm): #eqation number 13
    return P_blade_lm+P_fuselage_lm+P_induced_lm

def T_vfly_l(H,V_vfly_l): #eqation number 12
    return H/V_vfly_l

d_hfly_lm = None #horizontal distance between the UAV-IRS and the HP

def T_hfly_l(d_hfly_lm,V_hfly_lm): #eqation number 17
    return d_hfly_lm/V_hfly_lm

T_km_com=None # time required to complete the computation of the data
T_kml_up=None # time required to upload the data

def T_lm_hov(T_km_com,T_kml_up,T_ml_down): #eqation number 19
    return T_km_com+T_kml_up+T_ml_down
    #we have to take the max{T_com_km,Tup_kml}

#above all the function are define now we have to define the fitness function

#genetic algorithm to optimize the UAV-IRS assignment

import random

def initialize_population(population_size, uav_irs_options):
    population = []
    for _ in range(population_size):
        individual = {'uav_irs': random.choice(uav_irs_options)}
        fitness = random.uniform(1, 100)
        population.append((individual, fitness))
    return population

def evaluate_fitness(individual):
    return random.uniform(1, 100)  # Updated fitness evaluation function

def selection(population):
    sorted_population = sorted(population, key=lambda x: x[1])
    parent1 = sorted_population[0]
    parent2 = sorted_population[1]
    return parent1, parent2

def crossover(parent1, parent2, beta=0.5):
    offspring = {}
    for key in parent1[0].keys():
        offspring[key] = parent1[0][key] if parent1[1] < parent2[1] else parent2[0][key]
    offspring_fitness = beta * parent1[1] + (1 - beta) * parent2[1]
    return offspring, offspring_fitness

def mutation(offspring, uav_irs_options, p_mutation=0.5):
    for key in offspring.keys():
        if random.random() < p_mutation:
            offspring[key] = random.choice(uav_irs_options)
    return offspring

def genetic_algorithm(uav_irs_options, population_size, generations, p_mutation):
    population = initialize_population(population_size, uav_irs_options)
    
    for gen in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = selection(population)
            offspring, offspring_fitness = crossover(parent1, parent2, beta=0.5)
            offspring = mutation(offspring, uav_irs_options, p_mutation)
            offspring_fitness = evaluate_fitness(offspring)  # Recalculate fitness after mutation
            new_population.append((offspring, offspring_fitness))
        
        combined_population = population + new_population
        combined_population = sorted(combined_population, key=lambda x: x[1])[:population_size]
        population = combined_population
        
        best_individual, best_fitness = population[0]
        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")
        print(f"Best Individual: {best_individual}")
    
    print("\nBest individual after optimization:")
    print(f"UAV-IRS Assignment: {best_individual['uav_irs']}")
    print(f"Fitness: {best_fitness}")
    return best_individual

# Parameters
uav_irs_options = [f'UAV_IRS_{i+1}' for i in range(10)]
population_size = 50
generations = 20
p_mutation = 0.8

# Run the genetic algorithm
best_solution = genetic_algorithm(uav_irs_options, population_size, generations, p_mutation)
            