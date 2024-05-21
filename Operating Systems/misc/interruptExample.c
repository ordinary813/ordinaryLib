#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

// Signal handler function
void sigint_handler(int signum)
{
    printf("\nCaught SIGINT signal (Ctrl+C)\n");
    // Restore default signal handler for SIGINT
    signal(SIGINT, SIG_DFL);
    // Exit the program
    exit(signum);
}

int main()
{
    // Install signal handler for SIGINT
    signal(SIGINT, sigint_handler);

    printf("Press Ctrl+C to trigger SIGINT signal...\n");

    // Infinite loop to keep the program running
    while (1)
    {
        // Do nothing, just wait for the signal
        sleep(1);
    }

    return 0;
}