#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "socket.h"
#include <fstream>
#include <pthread.h>

#define HEADERSIZE    2
#define DATASIZE      1024
#define PACKETSIZE    HEADERSIZE + DATASIZE
#define MAXSEQNUM     64
#define MAXSSTHRESH   DATASIZE * MAXSEQNUM

void reliablyReceive(unsigned short int myUDPport, char* destinationFile)
{
    std::ofstream outfile(destinationFile, std::ofstream::binary); // output file stream
    char packet[PACKETSIZE];                    // buffer for receiving packets
    char data_buffer[MAXSSTHRESH];              // buffer for storing received data
    char ip[50];
    char port[6];
    int rec_bytes = 0;
    char lastConsecSeqNum = 63;                 // Tracks the last seqNum in a consecutive sequence of packets

    // Initialize UDP socket to listen on given port
    sprintf(port, "%d", (int) myUDPport);
    my::sockudp receiver;
    receiver.init_receive(port);

    while(1)
    {
        rec_bytes = receiver.receive(packet, ip, PACKETSIZE);
        if (rec_bytes > 0)
        {
//	    std::cout << packet << std::endl;
	    // packet is next packet in consecutive order after last consecutive set 
	    if((packet[0] == (lastConsecSeqNum + 1)) || ((packet[0] == 0) && (lastConsecSeqNum == 63)))
	    {
		lastConsecSeqNum = packet[0];
	    }
            // Send acknowledgement
            receiver.send(ip, port, &lastConsecSeqNum, 1);

	    // Buffer the data from the packet
            char seqNum = packet[0];
            int offset = ((int) seqNum) * DATASIZE;
            memcpy(data_buffer + offset, packet + HEADERSIZE, rec_bytes - HEADERSIZE);
            if((lastConsecSeqNum == 63) || (packet[1] == 'T'))
            {
                // write data buffer to file
                int bufferSize = (((int)lastConsecSeqNum) * DATASIZE) + rec_bytes - HEADERSIZE;
                outfile.write(data_buffer, bufferSize);
            }
	    // isEnd flag set to True
            if (packet[1] == 'T')
            {
		receiver.end_receive();
		outfile.close();
            	return;
            }
        }
    }
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
