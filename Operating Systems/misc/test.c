#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>

void main()
{
    pid_t x = fork();
    if (x < 0)
    {
        exit(1);
    }
    else if (x > 0)
    {
        printf("%d - father\n", getpid());
        exit(0);
    }
    else
    {
        char *const argv[] = {"sleep", "1", NULL};
        execv("/bin/sleep", argv);
        printf("%d - son\n", getpid());
    }
}