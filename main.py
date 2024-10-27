import random
import time
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Dijkstra's Algorithm to find the shortest path from a single source
def dijkstra(graph, start, n):
    distances = {node: float('inf') for node in range(n)}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

# Calculate the diameter of the graph
def calculate_diameter(graph, n):
    max_distance = 0
    for node in range(n):
        distances = dijkstra(graph, node, n)
        max_distance = max(max_distance, max(distances.values()))
    return max_distance

# Generate a random graph as an adjacency list
def generate_random_graph(n, density=0.01):
    graph = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:  # Add an edge with some probability
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight  # Undirected graph
    return graph

# Measure execution time for different input sizes
def measure_execution_time(n_values):
    experimental_results = []
    theoretical_results = [n**2 * np.log(n) for n in n_values]  # O(V^2 log V) theoretical complexity

    for n in n_values:
        print(f"Running diameter calculation for n = {n}...")
        graph = generate_random_graph(n)  # Sparse graph

        start_time = time.perf_counter_ns()  # Start time in nanoseconds
        diameter = calculate_diameter(graph, n)
        end_time = time.perf_counter_ns()  # End time in nanoseconds

        elapsed_time = end_time - start_time  # Execution time
        experimental_results.append(elapsed_time)
        print(f"Diameter: {diameter} | Execution Time: {elapsed_time} ns")

    return experimental_results, theoretical_results

# Calculate the scaling factor and adjusted theoretical results
def calculate_scaling_factor(experimental_results, theoretical_results):
    scaling_constants = [exp / theo for exp, theo in zip(experimental_results, theoretical_results)]
    avg_scaling_factor = np.mean(scaling_constants)
    adjusted_theoretical = [avg_scaling_factor * theo for theo in theoretical_results]
    return avg_scaling_factor, adjusted_theoretical

# Plot comparison between experimental and adjusted theoretical results
def plot_results(n_values, experimental_results, adjusted_theoretical):
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, experimental_results, label='Experimental Results', marker='o', color='blue', linestyle='-')
    plt.plot(n_values, adjusted_theoretical, label='Adjusted Theoretical Results', marker='x', color='red', linestyle='--')
    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.yscale('log')  # Logarithmic scale for y-axis
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (nanoseconds)')
    plt.title('Experimental vs Adjusted Theoretical Results')
    plt.legend()
    plt.grid(True)
    plt.xticks(n_values)  # Ensure ticks correspond to input sizes
    plt.show()

# Main function to run the performance analysis
def main():
    # Define input sizes
    n_values = [100,200,400,600,800,1000]  # Test with large n values

    # Measure execution times
    experimental_results, theoretical_results = measure_execution_time(n_values)

    # Calculate scaling factor and adjusted theoretical results
    avg_scaling_factor, adjusted_theoretical = calculate_scaling_factor(experimental_results, theoretical_results)

    # Print the numerical data
    print("\nNumerical Data:")
    print("n\tExperimental (ns)\tTheoretical\tScaling Constant\tAdjusted Theoretical")
    for i, n in enumerate(n_values):
        print(f"{n}\t{experimental_results[i]}\t\t{theoretical_results[i]}\t\t"
              f"{avg_scaling_factor:.4f}\t\t{adjusted_theoretical[i]}")

    # Plot the results
    plot_results(n_values, experimental_results, adjusted_theoretical)

# Run the main function
if __name__ == "__main__":
    main()
