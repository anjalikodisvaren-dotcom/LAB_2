# BSD3513 - LAB REPORT 2
# GENETIC ALGORITHM: 80-BIT PATTERN OPTIMISATION

import random
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. GENETIC ALGORITHM PARAMETERS

POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
N_GENERATIONS = 50

TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN
ELITISM = 2

# 2. FITNESS FUNCTION

def fitness(individual):
    """
    Calculate fitness based on the number of ones.

    Maximum fitness is 80 when the chromosome
    contains exactly 50 ones.
    """

    ones = int(np.sum(individual))
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# 3. INITIALISATION

def init_population():
    """
    Generate an initial population of 300 individuals.
    Each individual contains 80 binary values.
    """

    return np.random.randint(
        0, 2,
        size=(POP_SIZE, CHROM_LEN),
        dtype=np.int8
    )

# 4. SELECTION - TOURNAMENT SELECTION

def tournament_selection(population, fitness_scores):
    """
    Select three random individuals.
    The individual with the highest fitness wins.
    """

    candidates = np.random.randint(
        0, len(population),
        size=TOURNAMENT_K
    )

    winner = candidates[
        np.argmax(fitness_scores[candidates])
    ]
    return population[winner].copy()

# 5. CROSSOVER - SINGLE-POINT CROSSOVER

def single_point_crossover(parent1, parent2):
    """
    Exchange genetic material after one random
    crossover point.
    """

    if np.random.random() >= CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, CHROM_LEN)
    child1 = np.concatenate((
        parent1[:point],
        parent2[point:]
    ))

    child2 = np.concatenate((
        parent2[:point],
        parent1[point:]
    ))

    return child1, child2

# 6. MUTATION - BIT-FLIP MUTATION

def bit_flip_mutation(individual):
    """
    Each bit has a probability of 1/80
    of changing from 0 to 1 or 1 to 0.
    """

    mutation_mask = (
        np.random.random(CHROM_LEN) < MUTATION_RATE
    )

    individual[mutation_mask] = (
        1 - individual[mutation_mask]
    )

    return individual

# 7. MAIN GENETIC ALGORITHM

def evolve(population):

    history = {
        "Generation": [],
        "Best Fitness": [],
        "Average Fitness": [],
        "Worst Fitness": [],
        "Number of Ones": []
    }

    best_individual = None
    best_fitness = -1

    for generation in range(1, N_GENERATIONS + 1):

        # Step 1: Fitness evaluation
        fitness_scores = np.array([
            fitness(individual)
            for individual in population
        ])

        # Step 2: Identify the best individual
        best_index = int(np.argmax(fitness_scores))

        current_best = float(fitness_scores[best_index])

        # Keep the best solution found so far
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[best_index].copy()

        # Step 3: Record population statistics
        history["Generation"].append(generation)
        history["Best Fitness"].append(current_best)
        history["Average Fitness"].append(
            float(np.mean(fitness_scores))
        )
        history["Worst Fitness"].append(
            float(np.min(fitness_scores))
        )
        history["Number of Ones"].append(
            int(np.sum(population[best_index]))
        )

        # Step 4: Elitism
        elite_indices = np.argsort(fitness_scores)[-ELITISM:]

        new_population = [
            population[index].copy()
            for index in elite_indices
        ]

        # Step 5: Selection, crossover and mutation
        while len(new_population) < POP_SIZE:

            parent1 = tournament_selection(
                population, fitness_scores
            )
            parent2 = tournament_selection(
                population, fitness_scores
            )
            child1, child2 = single_point_crossover(
                parent1, parent2
            )
            child1 = bit_flip_mutation(child1)
            child2 = bit_flip_mutation(child2)
            new_population.extend([child1, child2])

        # Replace the old population
        population = np.array(
            new_population[:POP_SIZE],
            dtype=np.int8
        )

    return (
        best_individual,
        best_fitness,
        pd.DataFrame(history)
    )

# 8. STREAMLIT APPLICATION

st.set_page_config(
    page_title="Genetic Algorithm - 80 Bit Pattern",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Genetic Algorithm: 80-Bit Pattern Optimisation")

st.caption(
    "BSD3513 - Introduction to Artificial Intelligence | Lab Report 2"
)

# 9. PROBLEM CONFIGURATION

st.subheader("Problem Configuration")

with st.expander("Genetic Algorithm Parameters", expanded=True):

    st.markdown(
        f"""
**Population Size:** {POP_SIZE}  
**Chromosome Length:** {CHROM_LEN} bits  
**Target Number of Ones:** {TARGET_ONES}  
**Maximum Fitness:** {MAX_FITNESS}  
**Number of Generations:** {N_GENERATIONS}  
**Selection:** Tournament (k={TOURNAMENT_K})  
**Crossover:** Single-Point (rate={CROSSOVER_RATE})  
**Mutation:** Bit-Flip (rate={MUTATION_RATE:.4f} per bit)  
**Elitism:** Top {ELITISM} individuals preserved
"""
    )
 
# 10. RANDOM SEED

seed = st.number_input(
    "Random Seed (for reproducibility)",
    min_value=0,
    value=42,
    step=1
)

# 11. RUN GENETIC ALGORITHM

if st.button("Run Genetic Algorithm", type="primary"):

    # Set random seed
    random.seed(int(seed))
    np.random.seed(int(seed))

    # Generate initial population
    initial_population = init_population()

    # Execute Genetic Algorithm
    best_individual, best_score, results = evolve(
        initial_population
    )

    # Calculate chromosome information
    number_of_ones = int(np.sum(best_individual))
    number_of_zeros = CHROM_LEN - number_of_ones
    binary_pattern = "".join(
        str(int(bit)) for bit in best_individual
    )

    # 12. DISPLAY BEST SOLUTION
    
    st.divider()
    st.subheader("Best Individual Found")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Best Fitness",
        f"{best_score:.0f} / {MAX_FITNESS}"
    )

    col2.metric(
        "Number of Ones",
        number_of_ones
    )

    col3.metric(
        "Number of Zeros",
        number_of_zeros
    )

    st.write("**Best 80-Bit Chromosome:**")
    st.code(binary_pattern)

    # 13. FITNESS CONVERGENCE GRAPH
    
    st.subheader("Fitness Convergence Graph")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results["Generation"],
        results["Best Fitness"],
        label="Best Fitness",
        linewidth=2
    )

    ax.plot(
        results["Generation"],
        results["Average Fitness"],
        label="Average Fitness",
        linestyle="--"
    )

    ax.plot(
        results["Generation"],
        results["Worst Fitness"],
        label="Worst Fitness",
        linestyle=":"
    )

    ax.axhline(
        y=MAX_FITNESS,
        label="Maximum Fitness",
        linestyle="-."
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.set_title("Genetic Algorithm Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # 14. GENERATION STATISTICS
    
    st.subheader("Generation-by-Generation Results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # 15. RESULT VALIDATION
  
    st.subheader("Solution Validation")
    if best_score == MAX_FITNESS and number_of_ones == TARGET_ONES:

        first_generation = int(
            results.loc[
                results["Best Fitness"] == MAX_FITNESS,
                "Generation"
            ].iloc[0]
        )

        st.success(
            f"Optimal solution found! "
            f"Number of ones = {TARGET_ONES}, "
            f"Fitness = {MAX_FITNESS}. "
            f"First found at generation {first_generation}."
        )

    else:

        st.warning(
            "The optimal solution was not reached in this run. "
            "Try another random seed."
        )

    # 16. DOWNLOAD RESULTS
  
    st.subheader("Download Results")
    csv_data = results.to_csv(index=False)
    st.download_button(
        label="Download Generation Results",
        data=csv_data,
        file_name="GA_Generation_Results.csv",
        mime="text/csv"
    )

# 17. APPLICATION FOOTER

st.divider()
st.caption(
    "BSD3513 Lab Report 2 | Genetic Algorithm Implementation"
)