#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define BUFFER_SIZE 100
#define HISTORY_SIZE 64

int main(void)
{
    close(2);
    dup(1);
    char command[BUFFER_SIZE];
    int run_in_backgronud = 0;
    char history[HISTORY_SIZE][BUFFER_SIZE];
    int history_count = 0;

    while (1)
    {
        if (!run_in_backgronud)
        {
            wait(NULL);
        }
        fprintf(stdout, "my-shell> ");
        memset(command, 0, BUFFER_SIZE);
        fgets(command, BUFFER_SIZE, stdin);

        // remove newline in the end of the command
        size_t length = strlen(command);
        if (length > 0 && command[length - 1] == '\n')
        {
            command[length - 1] = '\0';
            length--;
        }

        // check for exit
        if (strncmp(command, "exit", 4) == 0 && (command[4] == '\0'))
        {
            break;
        }

        // check for "history"
        if (strncmp(command, "history", 7) == 0)
        {
            // add history and continue
            strncpy(history[history_count], command, BUFFER_SIZE - 1);
            if (history_count < HISTORY_SIZE)
            {
                history_count++;
            }
            // history functionality
            for (int i = history_count - 1; i >= 0; i--)
            {
                printf("%d %s\n", i + 1, history[i]);
            }
            continue;
        }

        // adding of the new command
        strncpy(history[history_count], command, BUFFER_SIZE - 1);
        if (history_count < HISTORY_SIZE)
        {
            history_count++;
        }

        // check for "&" in the end
        if (length > 1 && command[length - 1] == '&')
        {
            run_in_backgronud = 1;
            command[length - 1] = '\0';
            command[length - 2] = '\0';
            length -= 2;
        }
        else
        {
            run_in_backgronud = 0;
        }

        pid_t pid = fork();
        if (pid < 0)
        {
            perror("error");
            continue;
        }

        // in the child process we will run the command, in the foreground or in the background
        if (pid == 0)
        {
            // TEST THIS *****************************************************
            char *args[BUFFER_SIZE / 2 + 1];
            int i = 0;

            // tokenize the command string into arguments
            char *token = strtok(command, " ");
            while (token != NULL)
            {
                args[i] = token;
                token = strtok(NULL, " ");
                i++;
            }
            // last character is NULL
            args[i] = NULL;

            // execute the command
            if (execvp(args[0], args) == -1)
            {
                perror("error");
            }

            // making sure the child process terminates
            exit(1);
        }
        else
        {
            if (run_in_backgronud == 0)
            {
                // the parent will wait for the completion if "&" is not present
                wait(NULL);
            }
            else if (run_in_backgronud == 1)
            {
                fprintf(stdout, "[PID %d] running in the background.\n", pid);
            }
        }
    }

    return 0;
}