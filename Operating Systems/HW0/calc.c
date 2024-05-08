#include <stdio.h>
#include <string.h>

int main()
{
    char buffer[6];
    fgets(buffer, sizeof(buffer), stdin);

    int first, second, result;
    char op;

    printf("str:%s\n", buffer);

    while (strcmp("exit\0\0", buffer) != 0)
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
        fgets(buffer, sizeof(buffer), stdin);
    }
    return 0;
}