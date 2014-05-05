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

struct addrinfo *p;
int sockUDP_fd;

// Start up a UDP socket with the given port
void start_udp(char* hostname, unsigned short int port_int)
{
    sockUDP_fd =  -1;
    struct addrinfo hints, *servinfo;
    int rv;
    int yes = 1;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE; // use my IP

    char port[6];
    sprintf(port, "%d", port_int);
    if ((rv = getaddrinfo(hostname, port, &hints, &servinfo)) != 0)
    {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return;
    }

    // loop through all the results and bind to the first we can
    for(p = servinfo; p != NULL; p = p->ai_next)
    {
        if ((sockUDP_fd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1)
        {
            perror("listener: socket");
            continue;
        }

        //setsockopt(sockUDP_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int));

        //if (bind(sockUDP_fd, p->ai_addr, p->ai_addrlen) == -1) {
        //    close(sockUDP_fd);
        //    perror("listener: bind");
        //    continue;
        //}

        break;
    }

    if (p == NULL)
    {
        fprintf(stderr, "listener: failed to bind socket\n");
        return;
    }

    freeaddrinfo(servinfo);
}

int send_udp(char* message)
{
    int numbytes;
    if ((numbytes = sendto(sockUDP_fd, message, strlen(message), 0, p->ai_addr,
         p->ai_addrlen)) == -1)
    {
        perror("sender: sendto");
        return -1;
    }
    return numbytes;
}

void reliablyTransfer(char* hostname, unsigned short int hostUDPport, char* filename,
					  unsigned long long int bytesToTransfer)
{
	start_udp(hostname, hostUDPport);
    send_udp("Hello!"); 
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
