#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semaphoreA, semaphoreB, semaphoreC;

void* thread1Func(void* p) {
    while (1) {
        sem_wait(&semaphoreA);  // Wait for semaphoreA
        printf("A ");           // Print 'A'
        sem_post(&semaphoreB);  // Signal semaphoreB for 'B'
    }
}

void* thread2Func(void* p) {
    while (1) {
        sem_wait(&semaphoreB);  // Wait for semaphoreB
        sem_wait(&semaphoreB);  // Wait again to ensure two 'A's are printed
        printf("B ");           // Print 'B'
        sem_post(&semaphoreC);  // Signal semaphoreC for 'C'
    }
}

void* thread3Func(void* p) {
    while (1) {
        sem_wait(&semaphoreC);  // Wait for semaphoreC
        printf("C\n");          // Print 'C' and newline
        sem_post(&semaphoreA);  // Signal semaphoreA for the next 'A'
        sem_post(&semaphoreA);  // Signal semaphoreA again for the next 'A'
    }
}

int main() {
    pthread_t thread1, thread2, thread3;

    // Initialize semaphores
    sem_init(&semaphoreA, 0, 2);  // semaphoreA starts with 2 to allow two 'A's initially
    sem_init(&semaphoreB, 0, 0);  // semaphoreB starts with 0
    sem_init(&semaphoreC, 0, 0);  // semaphoreC starts with 0

    // Create threads
    pthread_create(&thread1, NULL, thread1Func, NULL);
    pthread_create(&thread2, NULL, thread2Func, NULL);
    pthread_create(&thread3, NULL, thread3Func, NULL);

    // Wait for threads to finish (which they won't in this case due to while(1))
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);

    // Destroy semaphores
    sem_destroy(&semaphoreA);
    sem_destroy(&semaphoreB);
    sem_destroy(&semaphoreC);

    return 0;
}
