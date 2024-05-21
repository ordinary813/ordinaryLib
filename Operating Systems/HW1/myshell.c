#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define BUFFER_SIZE 100
#define HISTORY_SIZE 25

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

        // Check if the command is "history"
        if (strncmp(command, "history", 7) == 0 && (command[7] == '\0' || isspace((unsigned char)command[7])))
        {
            for (int i = 0; i < history_count; i++)
            {
                printf("%d %s\n", i + 1, history[i]);
            }
            continue;
        }

        // Add command to history
        if (strncmp(command, "history", 7) != 0)
        {
            // Move existing history entries down
            for (int i = HISTORY_SIZE - 1; i > 0; i--)
            {
                strcpy(history[i], history[i - 1]);
            }

            // Add new command to history
            strncpy(history[0], command, BUFFER_SIZE - 1);
            if (history_count < HISTORY_SIZE)
            {
                history_count++;
            }
        }

        // check for "&" in the end
        if (length > 1 && command[length - 1] == '&')
        {
            run_in_backgronud = 1;
            command[length - 1] = '\0';
            command[length - 2] = '\0';
            length -= 2;
        }

        pid_t pid = fork();
        if (pid < 0)
        {
            perror("Fork Failed");
            continue;
        }

        // in the child process we will run the command, in the foreground or in the background
        if (pid == 0)
        {
            char *args[BUFFER_SIZE / 2 + 1]; // Array to hold command and arguments
            int i = 0;

            // Tokenize the command string into arguments
            char *token = strtok(command, " ");
            while (token != NULL)
            {
                args[i] = token;
                token = strtok(NULL, " ");
                i++;
            }
            args[i] = NULL; // Null-terminate the array of arguments

            // Execute the command
            if (execvp(args[0], args) == -1)
            {
                perror("Execution failed");
            }
            exit(EXIT_FAILURE);
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
                printf("[PID %d] running in the background.", pid);
                run_in_backgronud = 0;
            }
        }
    }

    return 0;
}