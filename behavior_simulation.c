#include "behavior_simulation.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Initialize the Behavior Simulation
BehaviorSimulation* init_behavior_simulation(uint64_t node_count, uint64_t interaction_threshold) {
    BehaviorSimulation* simulation = (BehaviorSimulation*)malloc(sizeof(BehaviorSimulation));
    if (!simulation) return NULL;

    simulation->nodes = (SimulationNode*)malloc(sizeof(SimulationNode) * node_count);
    if (!simulation->nodes) {
        free(simulation);
        return NULL;
    }

    for (uint64_t i = 0; i < node_count; i++) {
        simulation->nodes[i].id = i;
        simulation->nodes[i].energy = 100; // Default energy
        simulation->nodes[i].tasks_completed = 0;
    }

    simulation->node_count = node_count;
    simulation->interaction_threshold = interaction_threshold;

    printf("Behavior Simulation initialized with %lu nodes and interaction threshold %lu.\n", node_count, interaction_threshold);
    return simulation;
}

// Simulate Interactions Between Nodes
void simulate_interactions(BehaviorSimulation* simulation) {
    if (!simulation) return;

    printf("Simulating interactions...\n");
    for (uint64_t i = 0; i < simulation->node_count; i++) {
        for (uint64_t j = 0; j < simulation->node_count; j++) {
            if (i != j) {
                SimulationNode* node1 = &simulation->nodes[i];
                SimulationNode* node2 = &simulation->nodes[j];
                if (abs((int64_t)(node1->energy - node2->energy)) < simulation->interaction_threshold) {
                    uint64_t energy_exchange = (node1->energy - node2->energy) / 2;
                    node1->energy -= energy_exchange;
                    node2->energy += energy_exchange;
                    printf("Node %lu exchanged energy with Node %lu. New energies: %lu, %lu\n", node1->id, node2->id, node1->energy, node2->energy);
                }
            }
        }
    }
}

// Predict Emergent Behaviors
void predict_emergent_behavior(BehaviorSimulation* simulation) {
    if (!simulation) return;

    printf("Predicting emergent behaviors...\n");
    uint64_t total_energy = 0;
    for (uint64_t i = 0; i < simulation->node_count; i++) {
        total_energy += simulation->nodes[i].energy;
    }

    uint64_t average_energy = total_energy / simulation->node_count;
    printf("Average energy: %lu\n", average_energy);

    printf("Nodes above average energy: ");
    for (uint64_t i = 0; i < simulation->node_count; i++) {
        if (simulation->nodes[i].energy > average_energy) {
            printf("%lu ", simulation->nodes[i].id);
        }
    }
    printf("\n");
}

// Destroy the Behavior Simulation
void destroy_behavior_simulation(BehaviorSimulation* simulation) {
    if (simulation) {
        free(simulation->nodes);
        free(simulation);
        printf("Behavior Simulation destroyed.\n");
    }
}
