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

#define MAXBUFLEN 300

int sockUDP_fd;

// Start up the UDP socket with the given port
void start_udp(unsigned short int port_int)
{
    sockUDP_fd =  -1;
    struct addrinfo hints, *servinfo, *p;
    int rv;
    int yes = 1;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE; // use my IP

    char port[6];
    sprintf(port, "%d", port_int);
    if ((rv = getaddrinfo(NULL, port, &hints, &servinfo)) != 0)
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

        setsockopt(sockUDP_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int));

        if (bind(sockUDP_fd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockUDP_fd);
            perror("listener: bind");
            continue;
        }

        break;
    }

    if (p == NULL)
    {
        fprintf(stderr, "listener: failed to bind socket\n");
        return;
    }

    freeaddrinfo(servinfo);
}

int receive_udp(char * buffer)
{
    int numbytes;
    struct sockaddr_storage their_addr;
    socklen_t addr_len;

    if ((numbytes = recvfrom(sockUDP_fd, buffer, MAXBUFLEN - 1, 0,
        (struct sockaddr *) &their_addr, &addr_len)) == -1)
    {
        perror("recvfrom");
        return -1;
    }
    buffer[numbytes] = '\0';
    return numbytes;
}

void reliablyReceive(unsigned short int myUDPport, char* destinationFile)
{
    char buffer[MAXBUFLEN];
    start_udp(myUDPport);
    receive_udp(buffer);
    printf("%s\n", buffer);
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
