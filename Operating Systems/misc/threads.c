#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Number of threads to create
#define NUM_THREADS 5

// Function that each thread will execute
void *print_hello(void *threadid)
{
    // Cast the thread id argument to a long integer
    long tid = (long)threadid;
    printf("Hello from thread #%ld!\n", tid);
    pthread_exit(NULL); // Exit the thread
}

int main(int argc, char *argv[])
{
    pthread_t threads[NUM_THREADS]; // Array to hold thread identifiers
    int rc;                         // Variable to hold return code of pthread functions
    long t;                         // Loop index

    // Create threads
    for (t = 0; t < NUM_THREADS; t++)
    {
        printf("Main: creating thread %ld\n", t);
        // Create a thread that will run the print_hello function
        rc = pthread_create(&threads[t], NULL, print_hello, (void *)t);
        if (rc)
        {
            // If pthread_create returns a non-zero value, an error occurred
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (t = 0; t < NUM_THREADS; t++)
    {
        rc = pthread_join(threads[t], NULL); // Wait for thread t to finish
        if (rc)
        {
            // If pthread_join returns a non-zero value, an error occurred
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    printf("Main: program completed. Exiting.\n");
    pthread_exit(NULL); // Exit the main thread
}
