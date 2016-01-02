#include <stdio.h>
#include <stdlib.h>

int fib(int n)
{
        if (n == 1) {
            return 0;
        } 
        else if (n == 2) {
            return 1;
        } 
        else {
            return fib(n-1) + fib(n-2);
        }

}

int main (int argc, char const* argv[])
{
        int n = strtol(argv[1]);
        printf("%d", fib(n));

        /* return code */
        return 0;
}
