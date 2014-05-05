#include "socket.h"

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int my::sockudp::send( std::string their_addr, std::string message)
{
    return send(their_addr, DEFAULT_PORT, message);
}

int my::sockudp::send( std::string their_addr, std::string sendPort, std::string message)
{
	struct addrinfo hints, *servinfo, *p;
    int rv;
    int numbytes;
    int sockfd;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;

    if ((rv = getaddrinfo(their_addr.c_str(), sendPort.c_str(), &hints, &servinfo)) != 0) {
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


    if ((numbytes = sendto(sockfd, message.c_str(), message.length(), 0,
     p->ai_addr, p->ai_addrlen)) == -1) {
	    perror("talker: sendto");
	    exit(1);
	}

    freeaddrinfo(servinfo);

    //printf("talker: sent %d bytes to %s:%s\n", numbytes, their_addr.c_str(), sendPort.c_str());
    //std::cout << message << std::endl;


    close(sockfd);

    return 0;
}

bool my::sockudp::recOrTimeOut(std::string & message, std::string & ip, int ms){
    fd_set socks;
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = ms * 1000; 
    FD_ZERO(&socks);
    FD_SET(sockinfd, &socks);

    int rv = select(sockinfd + 1, &socks, NULL, NULL, &tv);

    
    if(rv == -1)
        return false;
    else if (rv == 0)
        return false;
    else if (FD_ISSET(sockinfd,&socks))
    {
        receive(message, ip);
        return true;
    }
    else
        return false;
}

void my::sockudp::receive(std::string & message, std::string & ip){
	struct sockaddr_storage their_addr;

    //std::cout << "listener: listening" << std::endl;
	socklen_t addr_len = sizeof their_addr;
	char buf[MAXBUFLEN];
	char s[MAXBUFLEN];
	int numbytes;

	char ipbuf[50];

    if ((numbytes = recvfrom(sockinfd, buf, MAXBUFLEN-1 , 0,
        (struct sockaddr *)&their_addr, &addr_len)) == -1) {
        perror("recvfrom");
        exit(1);
    }

    //printf("listener: got packet from %s\n",
        inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s);//);

    sprintf(ipbuf, "%s", inet_ntop(their_addr.ss_family,
            get_in_addr((struct sockaddr *)&their_addr),
            s, sizeof s)); 									// pass address along into "ip"

    ip = std::string(ipbuf);

    //printf("listener: packet is %d bytes long\n", numbytes);
    buf[numbytes] = '\0';
    //printf("listener: packet contains \"%s\"\n", buf);
    message = std::string(buf);
    

}

int my::sockudp::init_receive(std::string inport)
{
    struct addrinfo hints, *servinfo, *p;
    int rv;
    
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE; // use my IP

    if ((rv = getaddrinfo(NULL, inport.c_str(), &hints, &servinfo)) != 0) {
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
