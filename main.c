#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "node_core.h"
#include "mirrored_engine.h"
#include "kaleidoscope_engine.h"
#include "memory_graph.h"
#include "data_ingestion.h"

int main() {
    // Seed random number generator for simulations
    srand(time(NULL));

    printf("\n--- Initializing AI System Simulation ---\n");

    // Initialize Components
    Node* node = init_node(1);
    MemoryGraph* memory_graph = init_memory_graph(100);
    KaleidoscopeEngine* kaleidoscope_engine = init_kaleidoscope_engine(memory_graph);
    MirroredNetwork* mirrored_network = init_mirrored_network(10);
    DataIngestionLayer* ingestion_layer = init_data_ingestion_layer(50);

    // Load Disease Data
    ingest_disease_data(ingestion_layer, "disease_data.csv");

    printf("\n--- Simulation Begins ---\n");

    for (int cycle = 0; cycle < 10; cycle++) {
        printf("\n--- Cycle %d ---\n", cycle + 1);

        // Node requests help if stuck
        if (rand() % 3 == 0) {  // Simulate a 1 in 3 chance of the node being stuck
            request_help_from_mirrored_network(node, mirrored_network, "Unsolved disease model");
        }

        // Simulate interaction between components
        char sample_data[256];
        snprintf(sample_data, sizeof(sample_data), "Compound_%d", rand() % 100);
        simulate_compound_interaction(kaleidoscope_engine, sample_data);

        // Add random memory nodes
        if (rand() % 2 == 0) {
            char memory_entry[256];
            snprintf(memory_entry, sizeof(memory_entry), "Cycle %d insight: %s", cycle, sample_data);
            add_memory_node(memory_graph, memory_entry, rand() % 100);
        }

        // Simulate ingestion updates
        if (rand() % 2 == 0) {
            char data_update[256];
            snprintf(data_update, sizeof(data_update), "Cycle %d additional data", cycle);
            add_to_ingestion_memory(ingestion_layer, data_update);
        }
    }

    printf("\n--- Simulation Completed ---\n");

    // Cleanup
    destroy_node(node);
    destroy_memory_graph(memory_graph);
    destroy_kaleidoscope_engine(kaleidoscope_engine);
    destroy_mirrored_network(mirrored_network);
    destroy_data_ingestion_layer(ingestion_layer);

    return 0;
}
