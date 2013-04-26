/* 

http://www.cse.yorku.ca/~oz/hash.html 

Usage:

$ python 
>>> from ctypes import cdll
>>> djb2 = cdll.LoadLibrary("./libdjb2.so")
>>> print djb2.hash("hello world")

*/

unsigned long
hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}
