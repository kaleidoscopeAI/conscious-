#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kaleidoscope_engine.h"
#include "memory_graph.h"

// Initialize the Kaleidoscope Engine
KaleidoscopeEngine* init_kaleidoscope_engine(MemoryGraph* memory_graph) {
    KaleidoscopeEngine* engine = (KaleidoscopeEngine*)malloc(sizeof(KaleidoscopeEngine));
    engine->memory_graph = memory_graph;
    engine->insight_count = 0;
    printf("Kaleidoscope Engine initialized.\n");
    return engine;
}

// Process Data and Generate Insights
void process_task(KaleidoscopeEngine* engine, const char* task_data) {
    if (!engine || !task_data) {
        printf("Kaleidoscope Engine: Invalid task data.\n");
        return;
    }

    printf("Processing task: %s\n", task_data);

    // Simulate insight generation
    char insight[256];
    snprintf(insight, sizeof(insight), "Insight from task '%s'", task_data);

    // Add the insight to the Memory Graph
    add_memory_node(engine->memory_graph, insight, rand() % 100);
    engine->insight_count++;

    printf("Kaleidoscope Engine generated insight: %s\n", insight);
}

// Feedback Loop for Refining Insights
void refine_insights(KaleidoscopeEngine* engine, const char* feedback) {
    if (!engine || !feedback) {
        printf("Kaleidoscope Engine: Invalid feedback.\n");
        return;
    }

    printf("Refining insights based on feedback: %s\n", feedback);

    // Simulate refinement logic
    MemoryNode* current = engine->memory_graph->head;
    while (current) {
        if (rand() % 2 == 0) { // Example: Randomly mark nodes as refined
            current->relevance += 10;
            printf("Refined Node %lu: %s (New Relevance: %d)\n", current->id, current->data, current->relevance);
        }
        current = current->next;
    }
}

// Cleanup the Kaleidoscope Engine
void destroy_kaleidoscope_engine(KaleidoscopeEngine* engine) {
    if (engine) {
        printf("Destroying Kaleidoscope Engine.\n");
        free(engine);
    }
}
