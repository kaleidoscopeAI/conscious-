#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "memory_graph.h"

typedef struct MemoryNode {
    uint64_t id;
    char* data;
    int relevance;
    struct MemoryNode* next;
} MemoryNode;

typedef struct MemoryGraph {
    MemoryNode* head;
    uint64_t max_nodes;
    uint64_t current_nodes;
} MemoryGraph;

// Initialize the Memory Graph
MemoryGraph* init_memory_graph(uint64_t max_nodes) {
    MemoryGraph* graph = (MemoryGraph*)malloc(sizeof(MemoryGraph));
    graph->head = NULL;
    graph->max_nodes = max_nodes;
    graph->current_nodes = 0;
    printf("Memory Graph initialized with capacity for %lu nodes.\n", max_nodes);
    return graph;
}

// Add a Memory Node
void add_memory_node(MemoryGraph* graph, const char* data, int relevance) {
    if (!graph || graph->current_nodes >= graph->max_nodes) {
        printf("Memory Graph is full or uninitialized.\n");
        return;
    }

    MemoryNode* new_node = (MemoryNode*)malloc(sizeof(MemoryNode));
    new_node->id = graph->current_nodes + 1;
    new_node->data = strdup(data);
    new_node->relevance = relevance;
    new_node->next = graph->head;
    graph->head = new_node;
    graph->current_nodes++;

    printf("Memory Node %lu added: %s (Relevance: %d)\n", new_node->id, data, relevance);
}

// Retrieve Most Relevant Nodes
void get_top_relevant_nodes(MemoryGraph* graph, int count) {
    if (!graph || !graph->head) {
        printf("Memory Graph is empty or uninitialized.\n");
        return;
    }

    printf("Top %d relevant memory nodes:\n", count);
    MemoryNode* current = graph->head;
    for (int i = 0; i < count && current != NULL; i++) {
        printf("  Node %lu: %s (Relevance: %d)\n", current->id, current->data, current->relevance);
        current = current->next;
    }
}

// Cleanup the Memory Graph
void destroy_memory_graph(MemoryGraph* graph) {
    if (!graph) return;

    MemoryNode* current = graph->head;
    while (current) {
        MemoryNode* temp = current;
        current = current->next;
        free(temp->data);
        free(temp);
    }
    free(graph);
    printf("Memory Graph destroyed.\n");
}
