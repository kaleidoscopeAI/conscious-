#ifndef MEMORY_GRAPH_H
#define MEMORY_GRAPH_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Memory Node Structure
typedef struct MemoryNode {
    char* data;
    uint8_t importance;
    uint64_t timestamp;
    struct MemoryNode** connections;
    uint64_t connection_count;
} MemoryNode;

// Memory Graph Structure
typedef struct {
    MemoryNode** nodes;
    uint64_t node_count;
    uint64_t max_nodes;
} MemoryGraph;

// Function Prototypes
MemoryGraph* init_memory_graph(uint64_t max_nodes);
MemoryNode* create_memory_node(const char* data, uint8_t importance);
void connect_memory_nodes(MemoryNode* node1, MemoryNode* node2);
void add_memory_node(MemoryGraph* graph, MemoryNode* node);
void recall_memory(MemoryGraph* graph, const char* keyword);
void evolve_memory(MemoryGraph* graph);
void destroy_memory_graph(MemoryGraph* graph);

#endif // MEMORY_GRAPH_H

#include "memory_graph.h"

// Initialize the Memory Graph
MemoryGraph* init_memory_graph(uint64_t max_nodes) {
    MemoryGraph* graph = (MemoryGraph*)malloc(sizeof(MemoryGraph));
    if (!graph) return NULL;

    graph->nodes = (MemoryNode**)malloc(sizeof(MemoryNode*) * max_nodes);
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }

    graph->node_count = 0;
    graph->max_nodes = max_nodes;

    return graph;
}

// Create a Memory Node
MemoryNode* create_memory_node(const char* data, uint8_t importance) {
    MemoryNode* node = (MemoryNode*)malloc(sizeof(MemoryNode));
    if (!node) return NULL;

    node->data = strdup(data);
    node->importance = importance;
    node->timestamp = time(NULL);
    node->connections = NULL;
    node->connection_count = 0;

    return node;
}

// Connect Two Memory Nodes
void connect_memory_nodes(MemoryNode* node1, MemoryNode* node2) {
    if (!node1 || !node2) return;

    node1->connections = (MemoryNode**)realloc(node1->connections, sizeof(MemoryNode*) * (node1->connection_count + 1));
    node2->connections = (MemoryNode**)realloc(node2->connections, sizeof(MemoryNode*) * (node2->connection_count + 1));

    node1->connections[node1->connection_count++] = node2;
    node2->connections[node2->connection_count++] = node1;

    printf("Connected memory nodes: '%s' <-> '%s'\n", node1->data, node2->data);
}

// Add a Memory Node to the Graph
void add_memory_node(MemoryGraph* graph, MemoryNode* node) {
    if (!graph || graph->node_count >= graph->max_nodes || !node) return;

    graph->nodes[graph->node_count++] = node;
    printf("Added memory node: '%s'\n", node->data);
}

// Recall Memory by Keyword
void recall_memory(MemoryGraph* graph, const char* keyword) {
    if (!graph || !keyword) return;

    printf("Recalling memories related to keyword: '%s'\n", keyword);
    for (uint64_t i = 0; i < graph->node_count; i++) {
        if (strstr(graph->nodes[i]->data, keyword)) {
            printf("Memory Found: '%s' (Importance: %u)\n", graph->nodes[i]->data, graph->nodes[i]->importance);
        }
    }
}

// Evolve Memory Connections
void evolve_memory(MemoryGraph* graph) {
    if (!graph) return;

    printf("Evolving memory graph...\n");
    for (uint64_t i = 0; i < graph->node_count; i++) {
        MemoryNode* node = graph->nodes[i];
        if (node->importance < 5) {
            // Remove low-importance connections
            for (uint64_t j = 0; j < node->connection_count; j++) {
                node->connections[j] = NULL;
            }
            node->connection_count = 0;
            printf("Pruned low-importance node: '%s'\n", node->data);
        }
    }
}

// Destroy the Memory Graph
void destroy_memory_graph(MemoryGraph* graph) {
    if (graph) {
        for (uint64_t i = 0; i < graph->node_count; i++) {
            free(graph->nodes[i]->data);
            free(graph->nodes[i]->connections);
            free(graph->nodes[i]);
        }
        free(graph->nodes);
        free(graph);
    }
}
