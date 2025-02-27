#ifndef KALEIDOSCOPE_ENGINE_H
#define KALEIDOSCOPE_ENGINE_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "pattern_recognition.h"
#include "optimization.h"

// Kaleidoscope Engine Structure
typedef struct {
    PatternRecognizer* recognizer;
    Optimizer* optimizer;
} KaleidoscopeEngine;

// Function Prototypes
KaleidoscopeEngine* init_kaleidoscope_engine(void);
void process_task(KaleidoscopeEngine* engine, const char* task_data);
void destroy_kaleidoscope_engine(KaleidoscopeEngine* engine);

#endif // KALEIDOSCOPE_ENGINE_H

#include "kaleidoscope_engine.h"

// Initialize the Kaleidoscope Engine
KaleidoscopeEngine* init_kaleidoscope_engine(void) {
    KaleidoscopeEngine* engine = (KaleidoscopeEngine*)malloc(sizeof(KaleidoscopeEngine));
    if (!engine) return NULL;

    engine->recognizer = init_pattern_recognizer();
    engine->optimizer = init_optimizer();

    if (!engine->recognizer || !engine->optimizer) {
        destroy_kaleidoscope_engine(engine);
        return NULL;
    }

    return engine;
}

// Process a Task in the Kaleidoscope Engine
void process_task(KaleidoscopeEngine* engine, const char* task_data) {
    if (!engine || !task_data) return;

    printf("Processing task: %s\n", task_data);

    // Recognize patterns
    PatternResult* result = recognize_patterns(engine->recognizer, task_data);

    // Optimize the recognized patterns
    optimize_result(engine->optimizer, result);

    // Print the optimized results
    printf("Optimized task result: %s\n", result->optimized_data);

    // Clean up
    destroy_pattern_result(result);
}

// Destroy the Kaleidoscope Engine
void destroy_kaleidoscope_engine(KaleidoscopeEngine* engine) {
    if (engine) {
        if (engine->recognizer) destroy_pattern_recognizer(engine->recognizer);
        if (engine->optimizer) destroy_optimizer(engine->optimizer);
        free(engine);
    }
}
