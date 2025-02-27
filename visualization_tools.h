#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Visualization Metrics Structure
typedef struct {
    uint64_t* energy_levels;
    uint64_t* task_completions;
    uint64_t max_cycles;
    uint64_t current_cycle;
} VisualizationTools;

// Function Prototypes
VisualizationTools* init_visualization_tools(uint64_t max_cycles);
void update_metrics(VisualizationTools* tools, uint64_t energy, uint64_t tasks_completed);
void display_metrics(VisualizationTools* tools);
void destroy_visualization_tools(VisualizationTools* tools);

#endif // VISUALIZATION_TOOLS_H

#include "visualization_tools.h"

// Initialize Visualization Tools
VisualizationTools* init_visualization_tools(uint64_t max_cycles) {
    VisualizationTools* tools = (VisualizationTools*)malloc(sizeof(VisualizationTools));
    if (!tools) return NULL;

    tools->energy_levels = (uint64_t*)malloc(sizeof(uint64_t) * max_cycles);
    tools->task_completions = (uint64_t*)malloc(sizeof(uint64_t) * max_cycles);

    if (!tools->energy_levels || !tools->task_completions) {
        free(tools->energy_levels);
        free(tools->task_completions);
        free(tools);
        return NULL;
    }

    tools->max_cycles = max_cycles;
    tools->current_cycle = 0;

    return tools;
}

// Update Metrics
void update_metrics(VisualizationTools* tools, uint64_t energy, uint64_t tasks_completed) {
    if (!tools || tools->current_cycle >= tools->max_cycles) return;

    tools->energy_levels[tools->current_cycle] = energy;
    tools->task_completions[tools->current_cycle] = tasks_completed;

    printf("Cycle %lu: Energy=%lu, Tasks Completed=%lu\n",
           tools->current_cycle + 1, energy, tasks_completed);

    tools->current_cycle++;
}

// Display Metrics
void display_metrics(VisualizationTools* tools) {
    if (!tools) return;

    printf("\n--- Visualization Metrics ---\n");
    for (uint64_t i = 0; i < tools->current_cycle; i++) {
        printf("Cycle %lu: Energy=%lu, Tasks Completed=%lu\n",
               i + 1, tools->energy_levels[i], tools->task_completions[i]);
    }
    printf("--------------------------------\n");
}

// Destroy Visualization Tools
void destroy_visualization_tools(VisualizationTools* tools) {
    if (tools) {
        free(tools->energy_levels);
        free(tools->task_completions);
        free(tools);
    }
}
