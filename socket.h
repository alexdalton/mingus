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
		bool recOrTimeOut(std::string & message, std::string & ip, int ms);
		int send( std::string their_addr, std::string sendPort, std::string message);
		int send( std::string their_addr, std::string message);
		void receive(std::string & message, std::string & ip);
		int init_receive(std::string port);
		void end_receive();
	};
}

#endif
