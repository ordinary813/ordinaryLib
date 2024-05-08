#include <stdio.h>
#include <string.h>

int main()
{
    char buffer[6];
    fgets(buffer, sizeof(buffer), stdin);

    char exit[6];
    exit[0] = 'e';
    exit[1] = 'x';
    exit[2] = 'i';
    exit[3] = 't';
    exit[4] = '\n';
    exit[5] = 0;

    int first, second, result;
    char op;

    while (strcmp(exit, buffer) != 0)
    {
        first = buffer[0] - '0';
        second = buffer[4] - '0';
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
        fgets(buffer, sizeof(buffer), stdin);
    }
    return 0;
}