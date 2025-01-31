import random
import pandas as pd

def initialize_population(population_size, base_stations, uav_irs_options):
    population = []
    for _ in range(population_size):
        individual = {
            'base_station': random.choice(base_stations),
            'uav_irs': random.choice(uav_irs_options)
        }
        fitness = random.uniform(1, 100)
        population.append((individual, fitness))
    print(population)
    return population

def evaluate_fitness(individual):
    return random.uniform(1, 100)

def selection(population):
    sorted_population = sorted(population, key=lambda x: x[1])
    return sorted_population[:2]  # Select top two individuals

def crossover(parent1, parent2, beta=0.5):
    offspring = {}
    offspring['base_station'] = parent1[0]['base_station']
    offspring['uav_irs'] = parent1[0]['uav_irs'] if parent1[1] < parent2[1] else parent2[0]['uav_irs']
    offspring_fitness = beta * parent1[1] + (1 - beta) * parent2[1]
    return offspring, offspring_fitness

def mutation(offspring, uav_irs_options, assigned_uavs, p_mutation=0.5):
    if random.random() < p_mutation:
        available_uavs = list(set(uav_irs_options) - set(assigned_uavs.values()))
        if available_uavs:
            offspring['uav_irs'] = random.choice(available_uavs)
    return offspring

def genetic_algorithm(base_stations, uav_irs_options, population_size, generations, p_mutation):
    population = initialize_population(population_size, base_stations, uav_irs_options)
    assigned_uavs = {}
    all_results = []

    for gen in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = selection(population)
            offspring, offspring_fitness = crossover(parent1, parent2, beta=0.5)
            offspring = mutation(offspring, uav_irs_options, assigned_uavs, p_mutation)
            offspring_fitness = evaluate_fitness(offspring)
            new_population.append((offspring, offspring_fitness))
        
        combined_population = population + new_population
        combined_population = sorted(combined_population, key=lambda x: x[1])[:population_size]
        population = combined_population
        
        for ind in population:
            all_results.append((ind[0]['base_station'], ind[0]['uav_irs'], ind[1]))

        best_individual, best_fitness = population[0]
        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

    # Create DataFrame for all generations
    result_df = pd.DataFrame(all_results, columns=['Base Station', 'UAV-IRS Assignment', 'Fitness'])

    # Select final unique assignments for each base station
    final_data = []
    assigned_uavs = {}
    for ind in population:
        base_station = ind[0]['base_station']
        if base_station not in assigned_uavs and ind[0]['uav_irs'] not in assigned_uavs.values():
            assigned_uavs[base_station] = ind[0]['uav_irs']
            final_data.append((base_station, ind[0]['uav_irs'], ind[1]))

    final_assignment_df = pd.DataFrame(final_data, columns=['Base Station', 'Final UAV-IRS', 'Minimum Fitness'])

    return result_df, final_assignment_df

# Parameters
base_stations = [f'Base_Station_{i+1}' for i in range(10)]
uav_irs_options = [f'UAV_IRS_{i+1}' for i in range(10)]
population_size = 50
generations = 10
p_mutation = 0.5

# Run the genetic algorithm
result_df, final_assignment_df = genetic_algorithm(base_stations, uav_irs_options, population_size, generations, p_mutation)

print("\nFinal Assignment:")

print(final_assignment_df)
