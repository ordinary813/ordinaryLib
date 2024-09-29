/*
_________Longest substring_________
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int lengthOfLongestSubstring(char* s)
{
    int *currentAsciiCount = (int *) malloc (256 * sizeof(int));

    for(int i = 0; i < 256 ; i++)
    {
        currentAsciiCount[i] = 0;
    }

    int *lens = (int *) malloc(sizeof(s));
    for(int i = 0; s[i] != '\0'; i++)
    {
        
    }
}

int main(int argc, char *argv[])
{
    // Check if any arguments were provided
    if (argc < 2) {
        printf("No string provided.\n");
        return 1;
    }

    // Access the second command-line argument (argv[1])
    char *s = argv[1];

    // Remove the newline character if it exists
    size_t len = strlen(s);
    if (len > 0 && s[len - 1] == '\n') {
        s[len - 1] = '\0';
    }

    // Init ascii counter
    // char *s = "abcabcbb";
    int *currentAsciiCount = (int *) malloc (256 * sizeof(int));

    for(int i = 0; i < 256 ; i++)
    {
        currentAsciiCount[i] = 0;
    }

    int currentLen = 0;
    int maxLen = 0;

    for(int i = 0; s[i] != '\0'; i++)
    {
        currentLen = 0;
        for(int j = i + 1; s[j] != '\0'; j++)
        {
            if(currentAsciiCount[s[j]] == 0)
            {
                currentAsciiCount[s[j]]++;
                currentLen++;
            } else
            {
                break;
            }
            
        }

        if(maxLen < currentLen)
        {
            maxLen = currentLen;
        }
        
        memset(currentAsciiCount, 0, 256 * sizeof(int));
    }

    
    printf("max substring len is: %d\n", maxLen);
    return 0;
}