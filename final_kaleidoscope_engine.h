#ifndef FINAL_KALEIDOSCOPE_ENGINE_H
#define FINAL_KALEIDOSCOPE_ENGINE_H

#include "memory_graph.h"
#include <stdio.h>

// Final Kaleidoscope Engine Structure
typedef struct {
    MemoryGraph* memory_graph;
    int master_insight_count;
} FinalEngine;

// Function Prototypes
FinalEngine* init_final_engine(MemoryGraph* memory_graph);
void generate_master_insight(FinalEngine* engine, const char* meta_insight);
void destroy_final_engine(FinalEngine* engine);

// Debug Logging
#define DEBUG_LOG(fmt, ...) \
    do { \
        fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, __VA_ARGS__); \
    } while (0)

#endif // FINAL_KALEIDOSCOPE_ENGINE_H
