#include "global_ethics_system.h"
#include <stdlib.h>
#include <stdio.h>

// Initialize the Global Ethics System
GlobalEthicsSystem* init_global_ethics_system(uint8_t threshold) {
    GlobalEthicsSystem* system = (GlobalEthicsSystem*)malloc(sizeof(GlobalEthicsSystem));
    if (!system) return NULL;

    system->ethical_threshold = threshold;
    printf("Global Ethics System initialized with threshold %u.\n", threshold);
    return system;
}

// Evaluate an Action Based on Ethics
uint8_t evaluate_action(GlobalEthicsSystem* system, EthicalDecision* decision) {
    if (!system || !decision) return 0;

    // Compute overall ethical score
    uint8_t total_score = (decision->impact_on_individual +
                           decision->impact_on_collective +
                           decision->resource_fairness) / 3;

    printf("Evaluating ethical score: %u (Threshold: %u)\n", total_score, system->ethical_threshold);

    return total_score >= system->ethical_threshold;
}

// Enforce Ethical Guidelines on Tasks
void enforce_ethics(GlobalEthicsSystem* system, uint64_t node_id, EthicalDecision* decision) {
    if (!system || !decision) return;

    printf("Enforcing ethics on node %lu...\n", node_id);
    if (evaluate_action(system, decision)) {
        printf("Node %lu: Task approved (Ethical Score: %u).\n", node_id, system->ethical_threshold);
    } else {
        printf("Node %lu: Task rejected (Ethical Score: %u below threshold).\n", node_id, system->ethical_threshold);
    }
}

// Destroy the Global Ethics System
void destroy_global_ethics_system(GlobalEthicsSystem* system) {
    if (system) {
        free(system);
        printf("Global Ethics System destroyed.\n");
    }
}
