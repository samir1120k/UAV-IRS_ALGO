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
