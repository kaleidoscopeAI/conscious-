#ifndef BEHAVIOR_SIMULATION_H
#define BEHAVIOR_SIMULATION_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Node Structure for Simulation
typedef struct {
    uint64_t id;
    uint64_t energy;
    uint64_t tasks_completed;
} SimulationNode;

// Behavior Simulation Structure
typedef struct {
    SimulationNode* nodes;
    uint64_t node_count;
    uint64_t interaction_threshold;
} BehaviorSimulation;

// Function Prototypes
BehaviorSimulation* init_behavior_simulation(uint64_t node_count, uint64_t interaction_threshold);
void simulate_interactions(BehaviorSimulation* simulation);
void predict_emergent_behavior(BehaviorSimulation* simulation);
void run_simulation(BehaviorSimulation* simulation, uint64_t cycles);
void destroy_behavior_simulation(BehaviorSimulation* simulation);

#endif // BEHAVIOR_SIMULATION_H

#include "behavior_simulation.h"

// Initialize the Behavior Simulation
BehaviorSimulation* init_behavior_simulation(uint64_t node_count, uint64_t interaction_threshold) {
    printf("Initializing behavior simulation with %lu nodes and interaction threshold %lu\n", node_count, interaction_threshold);
    BehaviorSimulation* simulation = (BehaviorSimulation*)malloc(sizeof(BehaviorSimulation));
    if (!simulation) return NULL;

    simulation->nodes = (SimulationNode*)malloc(sizeof(SimulationNode) * node_count);
    if (!simulation->nodes) {
        free(simulation);
        return NULL;
    }

    for (uint64_t i = 0; i < node_count; i++) {
        simulation->nodes[i].id = i;
        simulation->nodes[i].energy = 100;
        simulation->nodes[i].tasks_completed = 0;
    }

    simulation->node_count = node_count;
    simulation->interaction_threshold = interaction_threshold;

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

// Run the Simulation for Multiple Cycles
void run_simulation(BehaviorSimulation* simulation, uint64_t cycles) {
    if (!simulation) return;

    for (uint64_t cycle = 0; cycle < cycles; cycle++) {
        printf("\n--- Simulation Cycle %lu ---\n", cycle + 1);
        simulate_interactions(simulation);
        predict_emergent_behavior(simulation);
    }
}

// Destroy the Behavior Simulation
void destroy_behavior_simulation(BehaviorSimulation* simulation) {
    if (simulation) {
        printf("Destroying behavior simulation\n");
        free(simulation->nodes);
        free(simulation);
    }
}
