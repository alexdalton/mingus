#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "socket.h"

void reliablyReceive(unsigned short int myUDPport, char* destinationFile)
{
    std::string message;
    std::string ip;
    my::sockudp receiver;
    char port[6];
    sprintf(port, "%d", (int) myUDPport);
    receiver.init_receive(port);
    receiver.receive(message, ip);
    receiver.end_receive();
    std::cout << message << std::endl;
    receiver.send(ip, port, "ACK");
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
