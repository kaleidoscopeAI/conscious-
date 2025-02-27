#include "learning_feedback.h"
#include <stdlib.h>
#include <stdio.h>

// Initialize the Learning Feedback System
LearningFeedbackSystem* init_learning_feedback_system(uint64_t max_scores, double learning_rate) {
    LearningFeedbackSystem* system = (LearningFeedbackSystem*)malloc(sizeof(LearningFeedbackSystem));
    if (!system) return NULL;

    system->feedback_scores = (double*)malloc(sizeof(double) * max_scores);
    if (!system->feedback_scores) {
        free(system);
        return NULL;
    }

    system->score_count = 0;
    system->max_scores = max_scores;
    system->learning_rate = learning_rate;

    printf("Learning Feedback System initialized with learning rate %.2f and capacity for %lu scores.\n", learning_rate, max_scores);
    return system;
}

// Record Feedback Score
void record_feedback(LearningFeedbackSystem* system, double score) {
    if (!system || system->score_count >= system->max_scores) {
        printf("Learning Feedback System: Cannot record score. Capacity reached or uninitialized system.\n");
        return;
    }

    system->feedback_scores[system->score_count++] = score;
    printf("Recorded feedback score: %.2f (Total scores: %lu)\n", score, system->score_count);
}

// Calculate Adjusted Learning Rate
double calculate_adjusted_learning_rate(LearningFeedbackSystem* system) {
    if (!system || system->score_count == 0) return system->learning_rate;

    double sum = 0.0;
    for (uint64_t i = 0; i < system->score_count; i++) {
        sum += system->feedback_scores[i];
    }
    double average_score = sum / system->score_count;

    // Adjust learning rate based on feedback trends
    double adjustment = 1.0 + (average_score - 50.0) / 100.0; // Normalize to +/- 50%
    double adjusted_rate = system->learning_rate * adjustment;

    printf("Adjusted learning rate: %.2f (Original: %.2f)\n", adjusted_rate, system->learning_rate);
    return adjusted_rate;
}

// Apply Learning Rate to a Model
void apply_learning_rate(double* learning_model, double adjustment) {
    if (!learning_model) return;

    *learning_model *= adjustment;
    printf("Learning model updated with adjustment: %.2f\n", adjustment);
}

// Destroy the Learning Feedback System
void destroy_learning_feedback_system(LearningFeedbackSystem* system) {
    if (system) {
        free(system->feedback_scores);
        free(system);
        printf("Learning Feedback System destroyed.\n");
    }
}
