#include <stdio.h>
#include <stdlib.h>
#include "membrane.h"

// Initialize the Membrane
Membrane* create_membrane(uint64_t total_energy, uint64_t max_nodes) {
    Membrane* membrane = (Membrane*)malloc(sizeof(Membrane));
    if (!membrane) {
        printf("Error: Failed to allocate memory for Membrane.\n");
        return NULL;
    }

    membrane->total_energy = total_energy;
    membrane->max_nodes = max_nodes;
    membrane->node_count = 0;
    membrane->nodes = (Node**)malloc(sizeof(Node*) * max_nodes);

    for (uint64_t i = 0; i < max_nodes; i++) {
        membrane->nodes[i] = NULL;
    }

    printf("Membrane initialized with total energy: %lu, max nodes: %lu\n", total_energy, max_nodes);
    return membrane;
}

// Add a Node to the Membrane
void add_node_to_membrane(Membrane* membrane, Node* node) {
    if (!membrane || !node) return;

    if (membrane->node_count >= membrane->max_nodes) {
        printf("Error: Membrane capacity reached. Cannot add Node %lu.\n", node->id);
        return;
    }

    membrane->nodes[membrane->node_count++] = node;
    printf("Node %lu added to the Membrane.\n", node->id);
}

// Redistribute Energy Across Nodes
void redistribute_energy(Membrane* membrane) {
    if (!membrane || membrane->node_count == 0) return;

    uint64_t available_energy = membrane->total_energy;
    uint64_t equal_share = available_energy / membrane->node_count;

    printf("Redistributing energy: Each node receives %lu units.\n", equal_share);

    for (uint64_t i = 0; i < membrane->node_count; i++) {
        if (membrane->nodes[i]) {
            membrane->nodes[i]->energy = equal_share;
            printf("Node %lu now has %lu energy.\n", membrane->nodes[i]->id, equal_share);
        }
    }
}

// Isolate a Failing Node
void isolate_node(Membrane* membrane, uint64_t node_id) {
    if (!membrane) return;

    for (uint64_t i = 0; i < membrane->node_count; i++) {
        if (membrane->nodes[i] && membrane->nodes[i]->id == node_id) {
            printf("Isolating Node %lu due to failure.\n", node_id);
            membrane->nodes[i]->is_active = 0; // Mark the node as inactive
            return;
        }
    }

    printf("Node %lu not found in the Membrane for isolation.\n", node_id);
}

// Replicate a Node
Node* replicate_node_in_membrane(Membrane* membrane, uint64_t parent_id, uint64_t new_id) {
    if (!membrane || membrane->node_count >= membrane->max_nodes) {
        printf("Error: Cannot replicate node. Membrane capacity reached.\n");
        return NULL;
    }

    Node* parent_node = NULL;
    for (uint64_t i = 0; i < membrane->node_count; i++) {
        if (membrane->nodes[i] && membrane->nodes[i]->id == parent_id) {
            parent_node = membrane->nodes[i];
            break;
        }
    }

    if (!parent_node || parent_node->energy < 50) {
        printf("Error: Parent Node %lu does not have enough energy to replicate.\n", parent_id);
        return NULL;
    }

    Node* new_node = init_node(new_id);
    new_node->energy = parent_node->energy / 2; // Split energy with the new node
    parent_node->energy /= 2;

    add_node_to_membrane(membrane, new_node);

    printf("Node %lu replicated into Node %lu. Energy split: Parent (%lu), Child (%lu).\n",
           parent_id, new_id, parent_node->energy, new_node->energy);

    return new_node;
}

// Clean Up Membrane
void destroy_membrane(Membrane* membrane) {
    if (!membrane) return;

    for (uint64_t i = 0; i < membrane->node_count; i++) {
        if (membrane->nodes[i]) {
            destroy_node(membrane->nodes[i]);
        }
    }

    free(membrane->nodes);
    free(membrane);
    printf("Membrane destroyed.\n");
}
