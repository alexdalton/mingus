#include "socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

void reliablyTransfer(char* hostname, unsigned short int hostUDPport, char* filename,
					  unsigned long long int bytesToTransfer)
{
	std::string host_str(hostname);
    std::string message("Hello!");
    char portbuf[6];
    sprintf(portbuf, "%d", (int) hostUDPport);
    std::string port(portbuf);
    my::sockudp sender;
    sender.send(host_str, portbuf, message);
    std::string buffer;
    std::string ip;
    sender.init_receive(portbuf);
    sender.receive(buffer, ip);
    std::cout << buffer << std::endl;
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
