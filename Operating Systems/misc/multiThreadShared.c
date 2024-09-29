#include <pthread.h>
#include <stdio.h>
#define N 1000
int i = 0;
int a[N];
void *f(void *arg)
{
    a[i] = i;
    i++;
    return NULL;
}
int main()
{
    while (1)
    {
        pthread_t threads[N];
        for (unsigned int i = 0; i < N; i++)
            pthread_create(&threads[i], NULL, f, NULL);
        for (unsigned int i = 0; i < N; i++)
            pthread_join(threads[i], NULL);
        printf("a[999] = %d\n", a[999]);
    }
    return 0;
}