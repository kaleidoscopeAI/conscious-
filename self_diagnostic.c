#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "self_diagnostic.h"

// Initialize the Diagnostic System
SelfDiagnosticSystem* init_self_diagnostic_system(uint64_t max_nodes) {
    SelfDiagnosticSystem* system = (SelfDiagnosticSystem*)malloc(sizeof(SelfDiagnosticSystem));
    system->node_health = (NodeHealth*)malloc(sizeof(NodeHealth) * max_nodes);
    system->max_nodes = max_nodes;
    system->node_count = 0;

    for (uint64_t i = 0; i < max_nodes; i++) {
        system->node_health[i].node_id = 0;
        system->node_health[i].energy = 0;
        system->node_health[i].is_active = 0;
        system->node_health[i].faults_detected = 0;
    }

    printf("Self-Diagnostic System initialized for %lu nodes.\n", max_nodes);
    return system;
}

// Register a Node for Monitoring
void register_node(SelfDiagnosticSystem* system, uint64_t node_id, uint64_t initial_energy) {
    if (!system || system->node_count >= system->max_nodes) {
        printf("Self-Diagnostic System: Cannot register node. Maximum capacity reached.\n");
        return;
    }

    system->node_health[system->node_count++] = (NodeHealth){
        .node_id = node_id,
        .energy = initial_energy,
        .is_active = 1,
        .faults_detected = 0
    };

    printf("Node %lu registered with initial energy %lu.\n", node_id, initial_energy);
}

// Check Node Health
int check_node_health(SelfDiagnosticSystem* system, uint64_t node_id) {
    if (!system) return 0;

    for (uint64_t i = 0; i < system->node_count; i++) {
        if (system->node_health[i].node_id == node_id) {
            NodeHealth* health = &system->node_health[i];

            // Simulate a fault check
            if (health->energy <= 10) {
                printf("Node %lu: Low energy detected (%lu). Marking as inactive.\n", node_id, health->energy);
                health->is_active = 0;
                health->faults_detected++;
                return 0; // Node is unhealthy
            }

            printf("Node %lu: Health check passed. Energy: %lu\n", node_id, health->energy);
            return 1; // Node is healthy
        }
    }

    printf("Node %lu not found in the diagnostic system.\n", node_id);
    return 0;
}

// Repair Node Faults
void repair_fault(SelfDiagnosticSystem* system, uint64_t node_id) {
    if (!system) return;

    for (uint64_t i = 0; i < system->node_count; i++) {
        if (system->node_health[i].node_id == node_id) {
            NodeHealth* health = &system->node_health[i];

            if (!health->is_active) {
                health->energy += 50; // Restore energy
                health->is_active = 1;
                printf("Node %lu: Fault repaired. Energy restored to %lu.\n", node_id, health->energy);
            } else {
                printf("Node %lu: No faults detected. No repair needed.\n", node_id);
            }
            return;
        }
    }

    printf("Node %lu not found for repair in the diagnostic system.\n", node_id);
}

// Generate System Health Report
void generate_health_report(SelfDiagnosticSystem* system) {
    if (!system) return;

    printf("\n--- System Health Report ---\n");
    for (uint64_t i = 0; i < system->node_count; i++) {
        NodeHealth* health = &system->node_health[i];
        printf("Node %lu: Energy = %lu, Active = %d, Faults Detected = %lu\n",
               health->node_id, health->energy, health->is_active, health->faults_detected);
    }
    printf("--- End of Report ---\n");
}

// Cleanup Diagnostic System
void destroy_self_diagnostic_system(SelfDiagnosticSystem* system) {
    if (system) {
        free(system->node_health);
        free(system);
        printf("Self-Diagnostic System destroyed.\n");
    }
}
