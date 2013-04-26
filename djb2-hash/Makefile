# http://ehuss.net/shared/

all:
	gcc -fPIC -g -c -Wall djb2.c 
	gcc -shared -o libdjb2.so djb2.o  -lc 

clean:
	@rm djb2.o


