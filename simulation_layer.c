#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "simulation_layer.h"

// Initialize Simulation Layer
SimulationLayer* init_simulation_layer(uint64_t max_simulations) {
    SimulationLayer* layer = (SimulationLayer*)malloc(sizeof(SimulationLayer));
    if (!layer) {
        printf("Error: Failed to allocate memory for Simulation Layer.\n");
        return NULL;
    }

    layer->simulations = (SimulationResult*)malloc(sizeof(SimulationResult) * max_simulations);
    layer->max_simulations = max_simulations;
    layer->current_simulations = 0;
    pthread_mutex_init(&layer->mutex, NULL); // Initialize mutex

    printf("Simulation Layer initialized with capacity for %lu simulations.\n", max_simulations);
    return layer;
}

// Run Simulation
void run_simulation(SimulationLayer* layer, const char* meta_insight) {
    if (!layer || !meta_insight) {
        printf("Simulation Layer: Invalid meta-insight.\n");
        return;
    }

    pthread_mutex_lock(&layer->mutex); // Lock mutex

    if (layer->current_simulations >= layer->max_simulations) {
        printf("Simulation Layer: Simulation queue full.\n");
        pthread_mutex_unlock(&layer->mutex); // Unlock mutex
        return;
    }

    printf("Running simulation for: %s\n", meta_insight);

    // Mock simulation results
    SimulationResult result;
    snprintf(result.description, sizeof(result.description), "Simulation Result for [%s]", meta_insight);
    result.is_valid = rand() % 2; // Randomly validate or reject

    layer->simulations[layer->current_simulations++] = result;

    pthread_mutex_unlock(&layer->mutex); // Unlock mutex

    if (result.is_valid) {
        printf("Simulation Output: Validated - %s\n", result.description);
    } else {
        printf("Simulation Output: Rejected - %s\n", result.description);
    }
}

// Retrieve Results
void get_simulation_results(SimulationLayer* layer) {
    if (!layer || layer->current_simulations == 0) {
        printf("Simulation Layer: No simulations to retrieve.\n");
        return;
    }

    pthread_mutex_lock(&layer->mutex); // Lock mutex

    printf("\n--- Simulation Results ---\n");
    for (uint64_t i = 0; i < layer->current_simulations; i++) {
        printf("Result %lu: %s (Valid: %s)\n", i + 1,
               layer->simulations[i].description,
               layer->simulations[i].is_valid ? "Yes" : "No");
    }
    printf("--- End of Results ---\n");

    pthread_mutex_unlock(&layer->mutex); // Unlock mutex
}

// Destroy Simulation Layer
void destroy_simulation_layer(SimulationLayer* layer) {
    if (layer) {
        pthread_mutex_destroy(&layer->mutex); // Destroy mutex
        free(layer->simulations);
        free(layer);
        printf("Simulation Layer destroyed.\n");
    }
}
