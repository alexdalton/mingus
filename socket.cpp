#include "socket.h"

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int my::sockudp::send( char* their_addr, char* sendPort, char* message, int length)
{
	struct addrinfo hints, *servinfo, *p;
    int rv;
    int numbytes;
    int sockfd;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;

    if ((rv = getaddrinfo(their_addr, sendPort, &hints, &servinfo)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return 1;
    }

    // loop through all the results and bind to the first we can
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1) {
            perror("talk: socket");
            continue;
        }

        break;
    }

    if (p == NULL) {
        fprintf(stderr, "talk: failed to bind socket\n");
        return 2;
    }

    if ((numbytes = sendto(sockfd, message, length, 0,
     p->ai_addr, p->ai_addrlen)) == -1) {
	    perror("talker: sendto");
	    exit(1);
	}

    freeaddrinfo(servinfo);

    close(sockfd);

    return 0;
}

int my::sockudp::recOrTimeOut(char* message, char* ip, int ms, int size){
    fd_set socks;
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = ms * 1000; 
    FD_ZERO(&socks);
    FD_SET(sockinfd, &socks);

    int rv = select(sockinfd + 1, &socks, NULL, NULL, &tv);

    
    if(rv == -1)
        return -1;
    else if (rv == 0)
        return -1;
    else if (FD_ISSET(sockinfd,&socks))
    {
        return receive(message, ip, size);
    }
    else
        return -1;
}

int my::sockudp::recOrTimeOutus(char* message, char* ip, int us, int size){
    fd_set socks;
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = us;
    FD_ZERO(&socks);
    FD_SET(sockinfd, &socks);

    int rv = select(sockinfd + 1, &socks, NULL, NULL, &tv);

    
    if(rv == -1)
        return -1;
    else if (rv == 0)
        return -1;
    else if (FD_ISSET(sockinfd,&socks))
    {
        return receive(message, ip, size);
    }
    else
        return -1;
}

int my::sockudp::receive(char * message, char * ip, int size){
	struct sockaddr_storage their_addr;

	socklen_t addr_len = sizeof their_addr;
	char s[MAXBUFLEN];
	int numbytes;

    if ((numbytes = recvfrom(sockinfd, message, size , 0,
        (struct sockaddr *)&their_addr, &addr_len)) == -1) {
        perror("recvfrom");
        return -1;
    }

    //printf("listener: got packet from %s\n",
        inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s);//);

    sprintf(ip, "%s", inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s)); 									// pass address along into "ip"

    return numbytes;
}

int my::sockudp::init_receive(char* inport)
{
    struct addrinfo hints, *servinfo, *p;
    int rv;
    
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE; // use my IP

    if ((rv = getaddrinfo(NULL, inport, &hints, &servinfo)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return 1;
    }

    // loop through all the results and bind to the first we can
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockinfd = socket(p->ai_family, p->ai_socktype,
                p->ai_protocol)) == -1) {
            perror("listener: socket");
            continue;
        }

        if (bind(sockinfd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sockinfd);
            perror("listener: bind");
            continue;
        }

        break;
    }

    if (p == NULL) {
        fprintf(stderr, "listener: failed to bind socket\n");
        return 2;
    }

    freeaddrinfo(servinfo);

    return 0;
}

void my::sockudp::end_receive()
{
	close(sockinfd);
}
