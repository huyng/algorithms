CC = g++
CCFLAGS = -fPIC -O3 -Wall -ffast-math -msse -msse2
LINKFLAGS = -shared -Wl

all: libconv.so

libconv.so: conv.cpp
	$(CC) -c $(CCFLAGS) $<
	$(CC) $(LINKFLAGS) -o $@ *.o

clean:
	rm *.so
	rm *.o

clean_tmp:
	rm *.o

.PHONY: all clean clean_tmp
	