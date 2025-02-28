#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "final_kaleidoscope_engine.h"
#include "memory_graph.h"

void debug(const char* message) {
    printf("[DEBUG]: %s\n", message);
}

// Initialize Final Kaleidoscope Engine
FinalEngine* init_final_engine(MemoryGraph* memory_graph) {
    debug("Initializing Final Kaleidoscope Engine.");
    FinalEngine* engine = (FinalEngine*)malloc(sizeof(FinalEngine));
    if (!engine) {
        printf("Error: Failed to allocate memory for Final Kaleidoscope Engine.\n");
        return NULL;
    }

    engine->memory_graph = memory_graph;
    engine->master_insight_count = 0;

    printf("Final Kaleidoscope Engine initialized.\n");
    return engine;
}

// Generate Master Insight
void generate_master_insight(FinalEngine* engine, const char* meta_insight) {
    debug("Generating master insight.");
    if (!engine || !meta_insight) {
        printf("Final Engine: Invalid meta-insight.\n");
        return;
    }

    printf("Processing meta-insight: %s\n", meta_insight);

    // Create a master insight
    char master_insight[512];
    snprintf(master_insight, sizeof(master_insight), "Master Insight: Abstracted from [%s]", meta_insight);

    // Store the master insight in the Memory Graph
    add_memory_node(engine->memory_graph, master_insight, rand() % 100);
    engine->master_insight_count++;

    printf("Generated Master Insight: %s\n", master_insight);
}

// Cleanup Final Kaleidoscope Engine
void destroy_final_engine(FinalEngine* engine) {
    debug("Destroying Final Kaleidoscope Engine.");
    if (engine) {
        printf("Destroying Final Kaleidoscope Engine.\n");
        free(engine);
    }
}
