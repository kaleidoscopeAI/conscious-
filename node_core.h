#ifndef NODE_CORE_H
#define NODE_CORE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Node Structure
typedef struct {
    uint64_t id;
    uint64_t energy;
    uint64_t memory_size;
    uint8_t is_active;
    uint8_t learning_drive;
    uint8_t growth_drive;
    uint8_t share_drive;
} Node;

// Function Prototypes
Node* init_node(uint64_t id);
void process_node(Node* node);
void replicate_node(Node* node);
void destroy_node(Node* node);

#endif // NODE_CORE_H

#include "node_core.h"

// Initialize a Node
Node* init_node(uint64_t id) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) return NULL;

    node->id = id;
    node->energy = 100;
    node->memory_size = 1024; // Default 1 KB
    node->is_active = 1;
    node->learning_drive = rand() % 10;
    node->growth_drive = rand() % 10;
    node->share_drive = rand() % 10;

    return node;
}

// Process a Node
void process_node(Node* node) {
    if (!node || !node->is_active) return;

    printf("Processing Node %lu\n", node->id);

    // Simulate learning
    if (node->learning_drive > 5) {
        printf("Node %lu is learning.\n", node->id);
        node->energy -= 10;
    }

    // Simulate sharing
    if (node->share_drive > 5) {
        printf("Node %lu is sharing knowledge.\n", node->id);
        node->energy -= 5;
    }

    // Simulate growth
    if (node->growth_drive > 7 && node->energy > 50) {
        printf("Node %lu is growing.\n", node->id);
        node->energy -= 20;
    }

    // Check energy
    if (node->energy <= 0) {
        node->is_active = 0;
        printf("Node %lu is inactive due to low energy.\n", node->id);
    }
}

// Replicate a Node
void replicate_node(Node* node) {
    if (!node || node->energy < 50) {
        printf("Node %lu does not have enough energy to replicate.\n", node->id);
        return;
    }

    printf("Node %lu is replicating.\n", node->id);
    node->energy -= 50;
}

// Destroy a Node
void destroy_node(Node* node) {
    if (node) {
        free(node);
    }
}
