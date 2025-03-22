#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "common.h"
#include <chrono>

bool validate_array(int* arr) {
    for (int i = 0; i < DATA_SIZE / sizeof(int); i++) {
        if (arr[i] != 262 * 262 * DIM / 2) {
            std::cerr << arr[i] << " for index " << i << std::endl;
            return false;
        }
    }

    return true;
}

int run_client(const std::string& ip_address, int iteration) {
    int sock;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) return 1;
    if (sock == -1) {
        return 1;
    }

    // Copypasta BEGIN
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, ip_address.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        close(sock);
        return 1;
    }

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        close(sock);
        return 1;
    }
    // Copypasta END

    int* matA = new int[DATA_SIZE / sizeof(int)];
    std::fill(matA, matA + DATA_SIZE / sizeof(int), 262);

    auto start = std::chrono::high_resolution_clock::now();
    sendAll(sock, matA, DATA_SIZE);

    int* matB = matA;
    sendAll(sock, matB, DATA_SIZE); // Just multiply by itself, input is immaterial.

    int* matC = matA;
    recvAll(sock, matC, DATA_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Network took: " << duration.count() << " seconds" << std::endl;

    if (!validate_array(matC))
        std::cout << "WRONG ANSWER!!" << std::endl;


    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: $0 <IP ADDRESS>" << std::endl;
    }
    if (run_client(argv[1], 1))  return 1; 
    return 0;
}