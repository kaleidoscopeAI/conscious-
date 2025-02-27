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

    layer->text_entries = (char**)malloc(sizeof(char*) * max_entries);
    layer->numerical_entries = (double**)malloc(sizeof(double*) * max_entries);
    layer->visual_entries = (char**)malloc(sizeof(char*) * max_entries);
    layer->max_entries = max_entries;
    layer->current_text = 0;
    layer->current_numerical = 0;
    layer->current_visual = 0;

    printf("Data Ingestion Layer initialized with capacity for %lu entries per type.\n", max_entries);
    return layer;
}

// Ingest Text Data
void ingest_text(DataIngestionLayer* layer, const char* text) {
    if (!layer || layer->current_text >= layer->max_entries) {
        printf("Error: Text ingestion capacity reached or layer uninitialized.\n");
        return;
    }

    layer->text_entries[layer->current_text] = strdup(text);
    printf("Ingested text data: %s\n", text);
    layer->current_text++;
}

// Ingest Numerical Data
void ingest_numerical(DataIngestionLayer* layer, const double* data, uint64_t count) {
    if (!layer || layer->current_numerical >= layer->max_entries) {
        printf("Error: Numerical ingestion capacity reached or layer uninitialized.\n");
        return;
    }

    double* entry = (double*)malloc(sizeof(double) * count);
    memcpy(entry, data, sizeof(double) * count);
    layer->numerical_entries[layer->current_numerical] = entry;

    printf("Ingested numerical data: [");
    for (uint64_t i = 0; i < count; i++) {
        printf("%lf%s", data[i], (i < count - 1) ? ", " : "");
    }
    printf("]\n");

    layer->current_numerical++;
}

// Ingest Visual Data
void ingest_visual(DataIngestionLayer* layer, const char* file_path) {
    if (!layer || layer->current_visual >= layer->max_entries) {
        printf("Error: Visual ingestion capacity reached or layer uninitialized.\n");
        return;
    }

    layer->visual_entries[layer->current_visual] = strdup(file_path);
    printf("Ingested visual data from file: %s\n", file_path);
    layer->current_visual++;
}

// Cleanup Data Ingestion Layer
void destroy_data_ingestion_layer(DataIngestionLayer* layer) {
    if (!layer) return;

    for (uint64_t i = 0; i < layer->current_text; i++) {
        free(layer->text_entries[i]);
    }
    for (uint64_t i = 0; i < layer->current_numerical; i++) {
        free(layer->numerical_entries[i]);
    }
    for (uint64_t i = 0; i < layer->current_visual; i++) {
        free(layer->visual_entries[i]);
    }

    free(layer->text_entries);
    free(layer->numerical_entries);
    free(layer->visual_entries);
    free(layer);

    printf("Data Ingestion Layer destroyed.\n");
}
