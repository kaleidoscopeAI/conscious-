#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "visualization_tools.h"

// Initialize Visualization Tools
VisualizationTools* init_visualization_tools() {
    VisualizationTools* tools = (VisualizationTools*)malloc(sizeof(VisualizationTools));
    if (!tools) {
        printf("Error: Failed to allocate memory for Visualization Tools.\n");
        return NULL;
    }

    tools->task_count = 0;
    tools->node_count = 0;
    printf("Visualization Tools initialized.\n");
    return tools;
}

// Update Node Metrics
void update_node_metrics(VisualizationTools* tools, Node* node) {
    if (!tools || !node) {
        printf("Visualization Tools: Invalid node or tools.\n");
        return;
    }

    printf("\n--- Node %lu Metrics ---\n", node->id);
    printf("Energy: %lu\n", node->energy);
    printf("Active: %s\n", node->is_active ? "Yes" : "No");
    printf("Tasks Completed: %lu\n", node->tasks_completed);
}

// Update Task Progress
void update_task_progress(VisualizationTools* tools, TaskManager* manager) {
    if (!tools || !manager) {
        printf("Visualization Tools: Invalid task manager or tools.\n");
        return;
    }

    printf("\n--- Task Progress ---\n");
    for (uint64_t i = 0; i < manager->task_count; i++) {
        printf("Task %lu: Priority %u, Assigned to Node %lu\n",
               manager->task_queue[i].task_id,
               manager->task_queue[i].priority,
               manager->task_queue[i].assigned_node_id);
    }
    printf("--- End of Task Progress ---\n");
}

// Display Insight Flow
void display_insight_flow(VisualizationTools* tools, const char* insight, const char* source, const char* destination) {
    if (!tools || !insight || !source || !destination) {
        printf("Visualization Tools: Invalid insight flow data.\n");
        return;
    }

    printf("\n--- Insight Flow ---\n");
    printf("Insight: %s\n", insight);
    printf("Source: %s\n", source);
    printf("Destination: %s\n", destination);
    printf("--- End of Insight Flow ---\n");
}

// Display Simulation Results
void display_simulation_results(VisualizationTools* tools, SimulationLayer* simulation_layer) {
    if (!tools || !simulation_layer) {
        printf("Visualization Tools: Invalid simulation layer or tools.\n");
        return;
    }

    printf("\n--- Simulation Results ---\n");
    for (uint64_t i = 0; i < simulation_layer->current_simulations; i++) {
        printf("Simulation %lu: %s (Valid: %s)\n", i + 1,
               simulation_layer->simulations[i].description,
               simulation_layer->simulations[i].is_valid ? "Yes" : "No");
    }
    printf("--- End of Simulation Results ---\n");
}

// Display System Status
void display_system_status(VisualizationTools* tools, Node** nodes, uint64_t node_count, TaskManager* manager) {
    if (!tools || !nodes || !manager) {
        printf("Visualization Tools: Invalid system components.\n");
        return;
    }

    printf("\n--- System Status ---\n");
    printf("Nodes: %lu\n", node_count);
    for (uint64_t i = 0; i < node_count; i++) {
        if (nodes[i]) {
            printf("Node %lu: Energy %lu, Active %s, Tasks Completed %lu\n",
                   nodes[i]->id, nodes[i]->energy,
                   nodes[i]->is_active ? "Yes" : "No",
                   nodes[i]->tasks_completed);
        }
    }

    printf("Tasks: %lu\n", manager->task_count);
    printf("--- End of System Status ---\n");
}

// Destroy Visualization Tools
void destroy_visualization_tools(VisualizationTools* tools) {
    if (tools) {
        free(tools);
        printf("Visualization Tools destroyed.\n");
    }
}
