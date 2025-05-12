#include <unistd.h>

#define DIM 1024
#define DATA_SIZE (DIM * DIM)
#define PORT 2626

ssize_t sendAll(int socket, void *buffer, size_t length);
ssize_t recvAll(int socket, void *buffer, size_t length);