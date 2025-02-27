#ifndef COLLABORATION_NODES_H
#define COLLABORATION_NODES_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "task_manager.h"
#include "node_core.h"

// Collaboration Node Structure
typedef struct {
    uint64_t id;
    Task** subtasks;
    uint64_t subtask_count;
    uint64_t max_subtasks;
    uint8_t is_active;
} CollaborationNode;

// Function Prototypes
CollaborationNode* init_collaboration_node(uint64_t id, uint64_t max_subtasks);
void assign_subtask(CollaborationNode* collab_node, Task* subtask);
void execute_subtasks(CollaborationNode* collab_node);
void destroy_collaboration_node(CollaborationNode* collab_node);

#endif // COLLABORATION_NODES_H

#include "collaboration_nodes.h"

// Initialize a Collaboration Node
CollaborationNode* init_collaboration_node(uint64_t id, uint64_t max_subtasks) {
    CollaborationNode* collab_node = (CollaborationNode*)malloc(sizeof(CollaborationNode));
    if (!collab_node) return NULL;

    collab_node->id = id;
    collab_node->subtasks = (Task**)malloc(sizeof(Task*) * max_subtasks);
    if (!collab_node->subtasks) {
        free(collab_node);
        return NULL;
    }

    collab_node->subtask_count = 0;
    collab_node->max_subtasks = max_subtasks;
    collab_node->is_active = 1;

    return collab_node;
}

// Assign a Subtask to the Collaboration Node
void assign_subtask(CollaborationNode* collab_node, Task* subtask) {
    if (!collab_node || collab_node->subtask_count >= collab_node->max_subtasks || !subtask) return;

    collab_node->subtasks[collab_node->subtask_count++] = subtask;
    printf("Subtask %lu assigned to Collaboration Node %lu.\n", subtask->task_id, collab_node->id);
}

// Execute Subtasks in the Collaboration Node
void execute_subtasks(CollaborationNode* collab_node) {
    if (!collab_node || collab_node->subtask_count == 0) return;

    printf("Executing subtasks in Collaboration Node %lu...\n", collab_node->id);

    for (uint64_t i = 0; i < collab_node->subtask_count; i++) {
        Task* subtask = collab_node->subtasks[i];
        if (subtask) {
            printf("Executing Subtask %lu (Priority: %u, Energy Required: %lu)...\n",
                   subtask->task_id, subtask->priority, subtask->energy_required);
            free(subtask);
        }
    }

    collab_node->subtask_count = 0; // Reset subtask count after execution
}

// Destroy the Collaboration Node
void destroy_collaboration_node(CollaborationNode* collab_node) {
    if (collab_node) {
        for (uint64_t i = 0; i < collab_node->subtask_count; i++) {
            free(collab_node->subtasks[i]);
        }
        free(collab_node->subtasks);
        free(collab_node);
    }
}
