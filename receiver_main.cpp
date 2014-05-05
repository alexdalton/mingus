#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "socket.h"
#include <fstream>
#include <pthread.h>

#define HEADERSIZE    3
#define DATASIZE      1024
#define PACKETSIZE    HEADERSIZE + DATASIZE
#define MAXSEQNUM     64
#define MAXSSTHRESH   DATASIZE * MAXSEQNUM

void reliablyReceive(unsigned short int myUDPport, char* destinationFile)
{
    std::ofstream outfile(destinationFile, std::ofstream::binary); // output file stream
    char packet[PACKETSIZE];                    // buffer for receiving packets
    char data_buffer[MAXSSTHRESH];              // buffer for storing received data
    char buffer_size = 0;
    char ip[50];
    char port[6];
    int rec_bytes = 0;
    int seqNumsReceived[MAXSEQNUM];             // boolean array to indicate which seqNums
                                                // were received already
    for(int i = 0; i < MAXSEQNUM; i++)
    {
        seqNumsReceived[i] = 0;
    }
    // Initialize UDP socket to listen on given port
    sprintf(port, "%d", (int) myUDPport);
    my::sockudp receiver;
    receiver.init_receive(port);
    while(1)
    {
        rec_bytes = receiver.receive(packet, ip, PACKETSIZE);
        if (rec_bytes > 0)
        {
            // Send acknowledgement
            receiver.send(ip, port, packet, 1);

            char seqNum = packet[0];
            int offset = (int) seqNum * DATASIZE;
            seqNumsReceived[(int) seqNum % MAXSEQNUM] = 1;
            memcpy(data_buffer + offset, packet + HEADERSIZE, rec_bytes - HEADERSIZE);
            int bufferFull = 0;
            int bufferSize = 0;
            for(int i; i < MAXSEQNUM; i++)
            {
                bufferFull += seqNumsReceived[i];
                if (seqNumsReceived[i])
                {
                    bufferSize += DATASIZE;
                }
            }
            if(bufferFull || packet[1] == 'T')
            {
                // write data buffer to file
                outfile.write(data_buffer, bufferSize);
                // clear out seqNumsReceived
                for(int i = 0; i < MAXSEQNUM; i++)
                {
                    seqNumsReceived[i] = 0;
                }
            }
        }
        // isEnd flag set to True
        if (packet[1] == 'T')
        {
            return;
        }
    }
    receiver.end_receive();
    outfile.close();
}

int main(int argc, char** argv)
{
	unsigned short int udpPort;
	
	if(argc != 3)
	{
		fprintf(stderr, "usage: %s UDP_port filename_to_write\n\n", argv[0]);
		exit(1);
	}
	
	udpPort = (unsigned short int)atoi(argv[1]);
	
	reliablyReceive(udpPort, argv[2]);
    return 0;
}
