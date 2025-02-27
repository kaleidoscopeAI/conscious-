#include <pthread.h>
#include "task_manager.h"

// Thread function for processing tasks
void* process_task_thread(void* arg) {
    TaskManager* manager = (TaskManager*)arg;
    Task* task;

    while ((task = get_next_task(manager)) != NULL) {
        // Process task in Primary Engine (or other modules)
        printf("Processing Task %lu: %s\n", task->task_id, task->task_data);
        // Simulate task execution
        usleep(1000); // Mock task duration
    }
    return NULL;
}

// Multi-threaded Task Processing
void process_tasks_multithreaded(TaskManager* manager, int thread_count) {
    pthread_t threads[thread_count];

    for (int i = 0; i < thread_count; i++) {
        pthread_create(&threads[i], NULL, process_task_thread, (void*)manager);
    }

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All tasks processed.\n");
}
