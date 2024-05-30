#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

int main()
{
    char buffer[6];
    fgets(buffer, sizeof(buffer), stdin);

    char exit[] = "exit\n";

    int first, second, result;
    char op;

    while (strcmp(exit, buffer) != 0)
    {
        if(isdigit(buffer[0]) && isspace(buffer[1]) && isprint(buffer[2]) && isspace(buffer[3]) && isdigit(buffer[4]))
        {
            first = atoi(&buffer[0]);
            second = atoi(&buffer[4]);
            op = buffer[2];

            switch (op)
            {
            case '+':
                result = first + second;
                break;
            case '-':
                result = first - second;
                break;
            case '*':
                result = first * second;
                break;
            case '/':
                result = first / second;
                break;
            default:
                break;
            }

            printf("%d\n", result);
            getchar();
        } else {
            printf("%s", buffer);

        }
        fgets(buffer, sizeof(buffer), stdin);
    }
    return 0;
}
