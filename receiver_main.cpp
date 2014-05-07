#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "socket.h"
#include <fstream>
#include <pthread.h>
#include <vector>
#include <algorithm>

#define HEADERSIZE    	 5
#define DATASIZE      	 1450
#define PACKETSIZE    	 HEADERSIZE + DATASIZE
#define NUMPACKSBUF      32
#define DATABUFFERSIZE   DATASIZE * NUMPACKSBUF


void reliablyReceive(unsigned short int myUDPport, char* destinationFile)
{
    std::ofstream outfile(destinationFile, std::ofstream::binary); // output file stream
    char packet[PACKETSIZE];                    // buffer to hold a received packet
    char ip[50];
    char port[6];
    int rec_bytes = 0;
    int lastConsecSeqNum = -1;                 // Tracks the last seqNum in a consecutive sequence of packets

    // Initialize UDP socket to listen on given port
    sprintf(port, "%d", (int) myUDPport);
    my::sockudp receiver;
    receiver.init_receive(port);

    bool receivedOtherPackets = false;
   //std::vector<int> packetTrack;
    while(1)
    {
        rec_bytes = receiver.receive(packet, ip, PACKETSIZE);
        if (rec_bytes > 0)
        {
	    // first 4 bytes are the seqNum integer
	    int seqNum;
	    memcpy(&seqNum, packet, 4);
	    // packet is next packet in consecutive order after last consecutive set
	    if(seqNum == (lastConsecSeqNum + 1))
	    {
		lastConsecSeqNum++;
		/*if (receivedOtherPackets)
		{
		    receivedOtherPackets = false;
		    std::sort (packetTrack.begin(), packetTrack.end());
		    for(std::vector<int>::iterator it=packetTrack.begin(); it != packetTrack.end(); it++)
		    {
			std::cout << *it << std::endl;
			if(*it == (lastConsecSeqNum + 1))
			{
				lastConsecSeqNum++;
			}
		    } 
		}*/
	    }
	    /*else
	    {
		std::cout << "Received other packet: " << seqNum << std::endl;
		packetTrack.clear();
		packetTrack.push_back(seqNum);
	    	receivedOtherPackets = true;
	    }*/
            // Send acknowledgement
            //std::cout << "ACK:" << lastConsecSeqNum << std::endl;
            receiver.send(ip, port, (char *) &lastConsecSeqNum, 4);

	    // write data to file in proper location
	    outfile.seekp(seqNum * DATASIZE ,outfile.beg);
            outfile.write(packet + HEADERSIZE, rec_bytes - HEADERSIZE);
            // isEnd flag set to True
            if ((packet[4] == 'T') && (lastConsecSeqNum == seqNum))
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
