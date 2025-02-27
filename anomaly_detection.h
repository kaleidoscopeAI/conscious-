#ifndef ANOMALY_DETECTION_H
#define ANOMALY_DETECTION_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Anomaly Detection Structure
typedef struct {
    double* data_points;
    uint64_t data_count;
    uint64_t max_data_points;
    double threshold;
} AnomalyDetector;

// Function Prototypes
AnomalyDetector* init_anomaly_detector(uint64_t max_data_points, double threshold);
void add_data_point(AnomalyDetector* detector, double value);
uint8_t detect_anomaly(AnomalyDetector* detector);
void destroy_anomaly_detector(AnomalyDetector* detector);

#endif // ANOMALY_DETECTION_H

#include "anomaly_detection.h"

// Initialize the Anomaly Detector
AnomalyDetector* init_anomaly_detector(uint64_t max_data_points, double threshold) {
    AnomalyDetector* detector = (AnomalyDetector*)malloc(sizeof(AnomalyDetector));
    if (!detector) return NULL;

    detector->data_points = (double*)malloc(sizeof(double) * max_data_points);
    if (!detector->data_points) {
        free(detector);
        return NULL;
    }

    detector->data_count = 0;
    detector->max_data_points = max_data_points;
    detector->threshold = threshold;

    return detector;
}

// Add a Data Point to the Detector
void add_data_point(AnomalyDetector* detector, double value) {
    if (!detector || detector->data_count >= detector->max_data_points) return;

    detector->data_points[detector->data_count++] = value;
    printf("Added data point: %.2f (Total points: %lu)\n", value, detector->data_count);
}

// Detect Anomalies Using Standard Deviation
uint8_t detect_anomaly(AnomalyDetector* detector) {
    if (!detector || detector->data_count < 2) {
        printf("Insufficient data for anomaly detection.\n");
        return 0;
    }

    // Calculate mean
    double sum = 0.0;
    for (uint64_t i = 0; i < detector->data_count; i++) {
        sum += detector->data_points[i];
    }
    double mean = sum / detector->data_count;

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (uint64_t i = 0; i < detector->data_count; i++) {
        variance_sum += pow(detector->data_points[i] - mean, 2);
    }
    double stddev = sqrt(variance_sum / detector->data_count);

    // Detect anomalies
    double last_point = detector->data_points[detector->data_count - 1];
    if (fabs(last_point - mean) > detector->threshold * stddev) {
        printf("Anomaly detected: %.2f (Threshold: %.2f)\n", last_point, detector->threshold * stddev);
        return 1; // Anomaly detected
    }

    printf("No anomaly detected.\n");
    return 0; // No anomaly
}

// Destroy the Anomaly Detector
void destroy_anomaly_detector(AnomalyDetector* detector) {
    if (detector) {
        free(detector->data_points);
        free(detector);
    }
}
