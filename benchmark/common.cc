#include "common.h"
#include <stdlib.h>
#include <unistd.h> 
#include <iostream>
#include <sys/socket.h> 

ssize_t sendAll(int socket, void *buffer_, size_t length) {
    char* buffer = (char*) buffer_;
    size_t totalSent=0;
    while (totalSent < length) {
        ssize_t sent = send(socket, buffer + totalSent, length - totalSent, 0);
        if (sent == -1) {
            perror("Something seriously wrong in sendAll");
            return -1;
        }
        totalSent += sent;
    }

    return totalSent;
}

ssize_t recvAll(int socket, void *buffer_, size_t length) {
    char* buffer = (char*) buffer_;
    size_t totalRcvd= 0;
    while (totalRcvd < length) {
        ssize_t rcvd = recv(socket, buffer + totalRcvd, length - totalRcvd, 0);
        if (rcvd <= 0) {
            perror("Something wrong in recvAll");
            return -1;
        }
        totalRcvd += rcvd;
    }
 
    return totalRcvd;
}