#include "socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <queue>

#define DATASIZE 1024
#define HEADERSIZE 3
#define PACKETSIZE HEADERSIZE + DATASIZE

volatile int sendingData = 1;
std::queue<char> ackQueue;

struct sendThreadParams
{
    char* hostname;
    unsigned short int port;
    char* filename;
    unsigned long long int bytesToTransfer;
};

void *send_function(void * inparams)
{
    sendThreadParams * params = (sendThreadParams *) inparams;
    std::ifstream in_file;
    in_file.open(params->filename, std::ifstream::binary);
    std::cout << params->hostname << params->port << params ->filename << params->bytesToTransfer;
    if (in_file)
    {
        char port_cstr[6];
        sprintf(port_cstr, "%d", (int) params->port);
        my::sockudp sender;
        int sentBytes = 0;      // number of reliably sent bytes from the file
        char windowSize = 1; 
        char seqNum = 0; // part of TCP header indicates sequence number
        char isEnd = 'F';       // part of TCP header flag indicates last packet        


        in_file.seekg(0, in_file.end);
        int length = in_file.tellg();
        in_file.seekg(0, in_file.beg);

        char packet[PACKETSIZE];
            // Set TCP data
        while(isEnd == 'F'){

            in_file.read(packet + HEADERSIZE, DATASIZE);

            // Read less bytes than data size, end of file reached
            if (in_file.gcount() < DATASIZE)
            {
                isEnd = 'T';
            }

            // Set TCP header
            packet[0] = seqNum;
            packet[1] = isEnd;

            // Send TCP packet
            sender.send(params->hostname, port_cstr, packet, in_file.gcount() + HEADERSIZE);

            
            seqNum = (seqNum + 1) % 64;
        }
        in_file.close();   
    }
   
    pthread_exit(NULL);
}

void *receive_function(void * inport)
{
    unsigned short int * port = (unsigned short int *) inport;
    char port_cstr[6];
    sprintf(port_cstr, "%d", (int) *port);
    char in_buffer[PACKETSIZE];
    char ip_buffer[50];
    int rec_bytes = 0;
    my::sockudp sender;
    sender.init_receive(port_cstr);
    while(sendingData)
    {
        rec_bytes = sender.receive(in_buffer, ip_buffer, PACKETSIZE);
        if (rec_bytes > 0)
        {
            for(int i = 0; i < rec_bytes; i++)
            {
                std::cout << "Acknolwedged: " << in_buffer[i] << std::endl;
                ackQueue.push(in_buffer[i]);
            }
        }
    }
}

void reliablyTransfer(char* hostname, unsigned short int hostUDPport, char* filename,
                      unsigned long long int bytesToTransfer)
{
    pthread_t send_thread, receive_thread;
    int iret1, iret2;

    // initialize params for sendThread
    sendThreadParams * sendParams = new sendThreadParams;
    sendParams->hostname = hostname;
    sendParams->port = hostUDPport;
    sendParams->filename = filename;
    sendParams->bytesToTransfer = bytesToTransfer;

    // create threads to send data and to receive ACKs
    iret1 = pthread_create(&send_thread, NULL, send_function, (void*) sendParams);
    iret2 = pthread_create(&receive_thread, NULL, receive_function, (void*) &hostUDPport);

    // wait until threads finish
    pthread_join(send_thread, NULL);
    pthread_join(receive_thread, NULL);
    delete sendParams;
}

int main(int argc, char** argv)
{
    unsigned short int udpPort;
    unsigned long long int numBytes;
    
    if(argc != 5)
    {
        fprintf(stderr, "usage: %s receiver_hostname receiver_port filename_to_xfer bytes_to_xfer\n\n", argv[0]);
        exit(1);
    }
    
    udpPort = (unsigned short int)atoi(argv[2]);
    numBytes = atoll(argv[4]);
    
    reliablyTransfer(argv[1], udpPort, argv[3], numBytes);
    return 0;
}
