#ifndef SELF_DIAGNOSTIC_H
#define SELF_DIAGNOSTIC_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Diagnostic State Structure
typedef struct {
    uint64_t node_id;
    uint8_t is_healthy;
    char* last_error;
} DiagnosticState;

// Self-Diagnostic System Structure
typedef struct {
    DiagnosticState* states;
    uint64_t node_count;
    uint64_t max_nodes;
} SelfDiagnosticSystem;

// Function Prototypes
SelfDiagnosticSystem* init_self_diagnostic_system(uint64_t max_nodes);
void check_node_health(SelfDiagnosticSystem* system, uint64_t node_id, uint8_t is_healthy, const char* error_message);
void repair_fault(SelfDiagnosticSystem* system, uint64_t node_id);
void display_diagnostic_report(SelfDiagnosticSystem* system);
void destroy_self_diagnostic_system(SelfDiagnosticSystem* system);

#endif // SELF_DIAGNOSTIC_H

#include "self_diagnostic.h"

// Initialize the Self-Diagnostic System
SelfDiagnosticSystem* init_self_diagnostic_system(uint64_t max_nodes) {
    SelfDiagnosticSystem* system = (SelfDiagnosticSystem*)malloc(sizeof(SelfDiagnosticSystem));
    if (!system) return NULL;

    system->states = (DiagnosticState*)malloc(sizeof(DiagnosticState) * max_nodes);
    if (!system->states) {
        free(system);
        return NULL;
    }

    for (uint64_t i = 0; i < max_nodes; i++) {
        system->states[i].node_id = i;
        system->states[i].is_healthy = 1;
        system->states[i].last_error = NULL;
    }

    system->node_count = 0;
    system->max_nodes = max_nodes;

    return system;
}

// Check Node Health
void check_node_health(SelfDiagnosticSystem* system, uint64_t node_id, uint8_t is_healthy, const char* error_message) {
    if (!system || node_id >= system->max_nodes) return;

    DiagnosticState* state = &system->states[node_id];
    state->is_healthy = is_healthy;

    if (!is_healthy && error_message) {
        state->last_error = strdup(error_message);
        printf("Node %lu reported an error: %s\n", node_id, error_message);
    } else {
        free(state->last_error);
        state->last_error = NULL;
        printf("Node %lu is healthy.\n", node_id);
    }
}

// Repair Fault
void repair_fault(SelfDiagnosticSystem* system, uint64_t node_id) {
    if (!system || node_id >= system->max_nodes) return;

    DiagnosticState* state = &system->states[node_id];
    if (!state->is_healthy) {
        printf("Repairing Node %lu...\n", node_id);
        free(state->last_error);
        state->last_error = NULL;
        state->is_healthy = 1;
        printf("Node %lu repaired and marked as healthy.\n", node_id);
    } else {
        printf("Node %lu is already healthy. No repair needed.\n", node_id);
    }
}

// Display Diagnostic Report
void display_diagnostic_report(SelfDiagnosticSystem* system) {
    if (!system) return;

    printf("\n--- Diagnostic Report ---\n");
    for (uint64_t i = 0; i < system->max_nodes; i++) {
        DiagnosticState* state = &system->states[i];
        printf("Node %lu: %s", state->node_id, state->is_healthy ? "Healthy" : "Faulty");
        if (!state->is_healthy && state->last_error) {
            printf(" (Error: %s)", state->last_error);
        }
        printf("\n");
    }
    printf("--------------------------\n");
}

// Destroy the Self-Diagnostic System
void destroy_self_diagnostic_system(SelfDiagnosticSystem* system) {
    if (system) {
        for (uint64_t i = 0; i < system->max_nodes; i++) {
            free(system->states[i].last_error);
        }
        free(system->states);
        free(system);
    }
}
