import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0) # Please do not remove this line
from preprocessing import load_data_from_file

def generate_random_value(bound = 10):
    return random.uniform(-bound/2, bound/2)

def create_individual(n=4):
    individual = [generate_random_value() for _ in range(n)]
    return individual 

features_x, sales_y = load_data_from_file()

def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_x.dot(theta)
    loss = np.multiply((y_hat - sales_y), (y_hat - sales_y)).mean()
    return loss

def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1 / (loss + 1)
    return fitness_value

def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()
    
    for i in range(len(individual1)):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]
    
    return individual1_new, individual2_new

def mutate(individual, mutation_rate=0.05):
    individual_m = individual.copy()
    for i in range(len(individual_m)):
        if random.random() < mutation_rate:
            individual_m[i] = generate_random_value()    
    return individual_m

def initialize_population(m):
    population = [create_individual() for _ in range(m)]
    return population

def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if index2 != index1:
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]
        
    return individual_s

def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)
    
    if gen % 1 == 0:
        print("Best loss:", compute_loss(sorted_population[m-1]), "with chromsome:", sorted_population[m-1])
        
    new_population = []
    while len(new_population) < m - elitism:
        # selection
        individual1 = selection(sorted_population, m)
        individual2 = selection(sorted_population, m)
        
        # crossover
        individual1, individual2 = crossover(individual1, individual2)
        
        # mutation
        individual1 = mutate(individual1)
        individual2 = mutate(individual2)
        
        new_population.append(individual1)
        new_population.append(individual2)
        
    # copy elitism chromosomes that have best fitness score to the next generation
    for ind in sorted_population[m-elitism:]:
        new_population.append(ind.copy())
    
    return new_population, compute_loss(sorted_population[m-1]) 

def run_genetic_algorithm():
    n_generations = 100
    m = 600
    population = initialize_population(m)
    losses_lst = []
    for i in range(n_generations):
        population, losses = create_new_population(population, 2, i)
        losses_lst.append(losses)
    return losses_lst

def visualize_loss(losses_lst):
    plt.plot(losses_lst, c='g')
    plt.xlabel('Generations')
    plt.ylabel('losses')
    plt.show()
    
def visualize_predict_gt():
    # visualization of ground truth and predict value
    population = initialize_population(m=100)
    sorted_population = sorted(population, key=compute_fitness)
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])

    estimated_prices = []
    for feature in features_x:
        estimated_price = sum(c*x for x, c in zip(feature, theta))
        estimated_prices.append(estimated_price)

    _, _ = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.plot(sales_Y, c='green', label='Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    individual = create_individual()
    print(individual) # [3.4442185152504816, 2.5795440294030243, -0.79428419169155, -2.4108324970703663]
    
    features_X , sales_Y = load_data_from_file()
    individual = [4.09, 4.82, 3.10, 4.02]
    fitness_score = compute_fitness(individual)
    print(fitness_score) # 1.0185991537088997e-06
    
    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    individual1, individual2 = crossover(individual1, individual2, 2.0)
    print ("individual1:", individual1) # individual1: [3.44, 2.57, -0.79, -2.41]
    print ("individual2:", individual2) # individual2: [4.09, 4.82, 3.1, 4.02]
    
    before_individual = [4.09, 4.82, 3.10, 4.02]
    after_individual = mutate(individual, mutation_rate=2.0)
    print(before_individual == after_individual) # False
    
    individual1 = [4.09, 4.82, 3.10, 4.02]
    individual2 = [3.44, 2.57, -0.79, -2.41]
    old_population = [individual1, individual2]
    new_population, _ = create_new_population(old_population, elitism=2, gen=1)
    
    # Visualize loss values
    losses_lst = run_genetic_algorithm()
    visualize_loss(losses_lst)
    
    # Visualize real sales price and estimated price
    visualize_predict_gt()