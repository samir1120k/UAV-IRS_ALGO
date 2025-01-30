import random
import math
import copy
import pandas as pd

def crossover(parent1, parent2, crossover_rate, N_l):
    offspring = {}
    if random.random() < crossover_rate:
        for bs in parent1:
            offspring[bs] = {}
            offspring[bs]['uav_irs'] = random.choice([parent1[bs]['uav_irs'], parent2[bs]['uav_irs']])
            beta = random.random()
            offspring[bs]['phase_shifts'] = [
                beta * parent1[bs]['phase_shifts'][i] + (1 - beta) * parent2[bs]['phase_shifts'][i]
                for i in range(N_l)
            ]
            for param in ['uplink_power', 'local_computation_time', 'uplink_time']:
                offspring[bs][param] = (
                    beta * parent1[bs][param] + (1 - beta) * parent2[bs][param]
                )
    else:
        offspring = copy.deepcopy(parent1)
    return offspring

def mutation(offspring, N_l, L, mutation_rate, parameter_bounds):
    for bs in offspring:
        if random.random() < mutation_rate:
            offspring[bs]['uav_irs'] = random.choice(L)
        for i in range(N_l):
            if random.random() < mutation_rate:
                mutation_step = random.uniform(-0.1, 0.1)
                offspring[bs]['phase_shifts'][i] += mutation_step
                offspring[bs]['phase_shifts'][i] %= 2 * math.pi
        for param in ['uplink_power', 'local_computation_time', 'uplink_time']:
            if random.random() < mutation_rate:
                mutation_step = random.uniform(-0.1, 0.1)
                offspring[bs][param] += mutation_step
                min_val, max_val = parameter_bounds[param]
                offspring[bs][param] = max(min(offspring[bs][param], max_val), min_val)
    return offspring

def tournament_selection(population, fitnesses, tournament_size):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = selected_indices[0]
    for idx in selected_indices:
        if fitnesses[idx] < fitnesses[best_index]:
            best_index = idx
    return copy.deepcopy(population[best_index])

def initialize_population(population_size, M, L, N_l, parameter_bounds):
    population = []
    for _ in range(population_size):
        individual = {}
        for bs in M:
            individual[bs] = {}
            individual[bs]['uav_irs'] = random.choice(L)
            individual[bs]['phase_shifts'] = [random.uniform(0, 2 * math.pi) for _ in range(N_l)]
            for param in ['uplink_power', 'local_computation_time', 'uplink_time']:
                min_val, max_val = parameter_bounds[param]
                individual[bs][param] = random.uniform(min_val, max_val)
        population.append(individual)
    return population

def cga_joint_optimization(M, L, population_size, generations, tournament_size, crossover_rate, mutation_rate, N_l, energy_consumption_function, parameter_bounds):
    population = initialize_population(population_size, M, L, N_l, parameter_bounds)
    fitnesses = [energy_consumption_function(individual) for individual in population]
    best_individual = copy.deepcopy(population[fitnesses.index(min(fitnesses))])
    best_fitness = min(fitnesses)

    for gen in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            offspring = crossover(parent1, parent2, crossover_rate, N_l)
            offspring = mutation(offspring, N_l, L, mutation_rate, parameter_bounds)
            new_population.append(offspring)

        fitnesses = [energy_consumption_function(individual) for individual in new_population]
        current_best_fitness = min(fitnesses)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = copy.deepcopy(new_population[fitnesses.index(best_fitness)])

        population = new_population

        print(f"Generation {gen+1}, Best Fitness: {best_fitness}")

    results = []
    for individual in population:
        individual_data = {}
        for bs in M:
            individual_data[f"{bs}_uav_irs"] = individual[bs]['uav_irs']
            individual_data[f"{bs}_phase_shifts"] = individual[bs]['phase_shifts']
            individual_data[f"{bs}_uplink_power"] = individual[bs]['uplink_power']
            individual_data[f"{bs}_local_computation_time"] = individual[bs]['local_computation_time']
            individual_data[f"{bs}_uplink_time"] = individual[bs]['uplink_time']
        individual_data["fitness"] = energy_consumption_function(individual)
        results.append(individual_data)

    df = pd.DataFrame(results)

    best_individual_data = {}
    for bs in M:
        best_individual_data[f"{bs}_uav_irs"] = best_individual[bs]['uav_irs']
        best_individual_data[f"{bs}_phase_shifts"] = best_individual[bs]['phase_shifts']
        best_individual_data[f"{bs}_uplink_power"] = best_individual[bs]['uplink_power']
        best_individual_data[f"{bs}_local_computation_time"] = best_individual[bs]['local_computation_time']
        best_individual_data[f"{bs}_uplink_time"] = best_individual[bs]['uplink_time']
    best_individual_data["fitness"] = energy_consumption_function(best_individual)
    best_individual_df = pd.DataFrame([best_individual_data])

    return best_individual_df, df

# Example usage
M = ['BS1', 'BS2', 'BS3']
L = ['UAV_IRS1', 'UAV_IRS2', 'UAV_IRS3']
parameter_bounds = {
    'uplink_power': (0.1, 1.0),
    'local_computation_time': (0.1, 10.0),
    'uplink_time': (0.1, 5.0)
}
population_size = 50
generations = 50
tournament_size = 5
crossover_rate = 0.8
mutation_rate = 0.1
N_l = 20

def energy_consumption_function(individual):
    energy = 0
    for bs in individual:
        energy += individual[bs]['uplink_power'] * individual[bs]['uplink_time']
    return energy

best_individual_df, population_df = cga_joint_optimization(
    M, L,
    population_size=population_size,
    generations=generations,
    tournament_size=tournament_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    N_l=N_l,
    energy_consumption_function=energy_consumption_function,
    parameter_bounds=parameter_bounds
)

print("Best Individual:")
print(best_individual_df)

#print("\nFinal Population:")
#print(population_df)

# best_individual_df.to_csv("best_individual.csv", index=False)
# population_df.to_csv("final_population.csv", index=False)