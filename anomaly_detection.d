../build/predictive/anomaly_detection.d \
 ../build/predictive/anomaly_detection.o: predictive/anomaly_detection.c \
 predictive/anomaly_detection.h

import std.stdio;

int[] detectAnomalies(int[] data, int threshold) {
    writeln("Input data: ", data); // Debug statement
    writeln("Threshold: ", threshold); // Debug statement
    
    int[] anomalies;
    foreach (num; data) {
        if (num > threshold) {
            writeln("Anomaly detected: ", num); // Debug statement
            anomalies ~= num;
        }
    }
    writeln("Total anomalies: ", anomalies.length); // Debug statement
    return anomalies;
}
