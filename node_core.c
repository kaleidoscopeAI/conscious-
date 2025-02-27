#include "node_core.h"
#include "mirrored_engine.h"
#include "kaleidoscope_engine.h"
#include <stdio.h>

// Node requests help from the Mirrored Network
void request_help_from_mirrored_network(Node* node, MirroredNetwork* network, const char* problem_description) {
    if (!node || !network) return;

    printf("Node %lu requesting help: %s\n", node->id, problem_description);
    generate_computational_suggestion(network, problem_description);
    propose_to_node(network, node);
}

// Process suggestions from Mirrored Network
void process_suggestion(KaleidoscopeEngine* engine, Suggestion* suggestion) {
    if (!engine || !suggestion) return;

    printf("Kaleidoscope Engine processing suggestion: %s\n", suggestion->description);
    simulate_compound_interaction(engine, suggestion->context);
}
