#ifndef MEMBRANE_H
#define MEMBRANE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "node_core.h"

// Membrane Structure
typedef struct {
    Node** nodes;
    uint64_t total_nodes;
    uint64_t energy_pool;
} Membrane;

// Function Prototypes
Membrane* create_membrane(uint64_t initial_energy, uint64_t max_nodes);
void distribute_energy(Membrane* membrane);
void replicate_nodes(Membrane* membrane);
void destroy_membrane(Membrane* membrane);

#endif // MEMBRANE_H

#include "membrane.h"

// Create a Membrane
Membrane* create_membrane(uint64_t initial_energy, uint64_t max_nodes) {
    Membrane* membrane = (Membrane*)malloc(sizeof(Membrane));
    if (!membrane) return NULL;

    membrane->nodes = (Node**)malloc(sizeof(Node*) * max_nodes);
    if (!membrane->nodes) {
        free(membrane);
        return NULL;
    }

    for (uint64_t i = 0; i < max_nodes; i++) {
        membrane->nodes[i] = NULL;
    }

    membrane->total_nodes = 0;
    membrane->energy_pool = initial_energy;

    return membrane;
}

// Distribute Energy to Nodes
void distribute_energy(Membrane* membrane) {
    if (!membrane || membrane->total_nodes == 0) return;

    uint64_t energy_per_node = membrane->energy_pool / membrane->total_nodes;

    for (uint64_t i = 0; i < membrane->total_nodes; i++) {
        if (membrane->nodes[i]) {
            membrane->nodes[i]->energy += energy_per_node;
            membrane->energy_pool -= energy_per_node;
        }
    }

    printf("Distributed %lu energy per node.\n", energy_per_node);
}

// Replicate Nodes
void replicate_nodes(Membrane* membrane) {
    for (uint64_t i = 0; i < membrane->total_nodes; i++) {
        Node* node = membrane->nodes[i];
        if (node && node->energy > 80) {
            Node* new_node = init_node(membrane->total_nodes + 1);
            if (new_node) {
                membrane->nodes[membrane->total_nodes++] = new_node;
                node->energy -= 50;
                printf("Node %lu replicated successfully.\n", node->id);
            }
        }
    }
}

// Destroy the Membrane
void destroy_membrane(Membrane* membrane) {
    if (!membrane) return;

    for (uint64_t i = 0; i < membrane->total_nodes; i++) {
        destroy_node(membrane->nodes[i]);
    }

    free(membrane->nodes);
    free(membrane);
}
