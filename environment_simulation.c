#include "environment_simulation.h"
#include <stdlib.h>
#include <stdio.h>

// Initialize the Environment Simulation
EnvironmentSimulation* init_environment_simulation(uint64_t max_states) {
    printf("Initializing Environment Simulation with max_states=%lu\n", max_states);
    EnvironmentSimulation* simulation = (EnvironmentSimulation*)malloc(sizeof(EnvironmentSimulation));
    if (!simulation) {
        printf("Failed to allocate memory for EnvironmentSimulation\n");
        return NULL;
    }

    simulation->states = (EnvironmentState*)malloc(sizeof(EnvironmentState) * max_states);
    if (!simulation->states) {
        printf("Failed to allocate memory for EnvironmentState array\n");
        free(simulation);
        return NULL;
    }

    simulation->state_count = 0;
    simulation->max_states = max_states;

    printf("Environment Simulation initialized with capacity for %lu states.\n", max_states);
    return simulation;
}

// Update the Environment State
void update_environment_state(EnvironmentSimulation* simulation, uint64_t data, uint64_t conditions) {
    printf("Updating Environment State with data=%lu, conditions=%lu\n", data, conditions);
    if (!simulation) {
        printf("Simulation is NULL\n");
        return;
    }
    if (simulation->state_count >= simulation->max_states) {
        printf("Reached maximum state capacity\n");
        return;
    }

    EnvironmentState* state = &simulation->states[simulation->state_count++];
    state->external_data = data;
    state->external_conditions = conditions;
    state->interaction_count = 0;

    printf("Environment State Updated: Data=%lu, Conditions=%lu\n", data, conditions);
}

// Simulate Interaction with the Environment
void simulate_interaction(EnvironmentSimulation* simulation, uint64_t node_id) {
    printf("Simulating interaction for node_id=%lu\n", node_id);
    if (!simulation) {
        printf("Simulation is NULL\n");
        return;
    }
    if (simulation->state_count == 0) {
        printf("No states available to interact with\n");
        return;
    }

    EnvironmentState* current_state = &simulation->states[simulation->state_count - 1];
    current_state->interaction_count++;

    printf("Node %lu interacting with environment: Data=%lu, Conditions=%lu, Interactions=%lu\n",
           node_id, current_state->external_data, current_state->external_conditions, current_state->interaction_count);
}

// Display the Environment State
void display_environment(EnvironmentSimulation* simulation) {
    printf("Displaying Environment States\n");
    if (!simulation) {
        printf("Simulation is NULL\n");
        return;
    }

    printf("\n--- Environment States ---\n");
    for (uint64_t i = 0; i < simulation->state_count; i++) {
        EnvironmentState* state = &simulation->states[i];
        printf("State %lu: Data=%lu, Conditions=%lu, Interactions=%lu\n",
               i + 1, state->external_data, state->external_conditions, state->interaction_count);
    }
    printf("---------------------------\n");
}

// Destroy the Environment Simulation
void destroy_environment_simulation(EnvironmentSimulation* simulation) {
    printf("Destroying Environment Simulation\n");
    if (simulation) {
        free(simulation->states);
        free(simulation);
        printf("Environment Simulation destroyed.\n");
    } else {
        printf("Simulation is NULL\n");
    }
}
