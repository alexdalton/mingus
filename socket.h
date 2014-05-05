#ifndef SOCKET_H
#define SOCKET_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string>
#include <iostream>
#define MAXBUFLEN 512
#define DEFAULT_PORT "2500"

namespace my{
	class sockudp { // udp
		//int bind(); // return fd
		int fd;
		int sockinfd;
	public:
		bool recOrTimeOut(char* message, char* ip, int ms, int size);
		int send( char* their_addr, char* sendPort, char* message, int length);
		int receive(char* message, char* ip, int size);
		int init_receive(char* inport);
		void end_receive();
	};
}

#endif
