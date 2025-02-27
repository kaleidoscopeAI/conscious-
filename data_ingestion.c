#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_ingestion.h"

// Initialize the Data Ingestion Layer
DataIngestionLayer* init_data_ingestion_layer(uint64_t max_entries) {
    DataIngestionLayer* layer = (DataIngestionLayer*)malloc(sizeof(DataIngestionLayer));
    if (!layer) {
        printf("Error: Failed to allocate memory for Data Ingestion Layer.\n");
        return NULL;
    }

    layer->binary_entries = (char**)malloc(sizeof(char*) * max_entries);
    if (!layer->binary_entries) {
        free(layer);
        printf("Error: Failed to allocate memory for binary entries.\n");
        return NULL;
    }

    layer->max_entries = max_entries;
    layer->current_count = 0;

    printf("Data Ingestion Layer initialized with capacity for %lu entries.\n", max_entries);
    return layer;
}

// Ingest Data
void ingest_data(DataIngestionLayer* layer, const char* text) {
    if (!layer || layer->current_count >= layer->max_entries) {
        printf("Error: Data Ingestion Layer is full or uninitialized.\n");
        return;
    }

    // Convert text to binary
    char* binary_entry = to_binary(text);
    if (!binary_entry) {
        printf("Error: Failed to convert text to binary.\n");
        return;
    }

    layer->binary_entries[layer->current_count++] = binary_entry;
    printf("Ingested Data: %s\n", text);
    printf("Stored as Binary: %s\n", binary_entry);
}

// Free Ingestion Layer
void free_ingestion_layer(DataIngestionLayer* layer) {
    if (!layer) return;

    for (uint64_t i = 0; i < layer->current_count; i++) {
        free(layer->binary_entries[i]);
    }
    free(layer->binary_entries);
    free(layer);

    printf("Data Ingestion Layer destroyed.\n");
}
