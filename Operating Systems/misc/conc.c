#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(void)
{
    int counter = 4;
    pid_t pid = fork();

    if (pid == 0)
    {
        counter += 5;
        printf("S - Counter is: %d\n", counter);
    }
    else
    {
        counter += 2;
        printf("P - Counter is: %d\n", counter);
    }
    return 0;
}