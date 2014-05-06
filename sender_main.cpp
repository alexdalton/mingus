#include "socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <queue>
#include <vector>

#define DATASIZE 1024
#define HEADERSIZE 2
#define PACKETSIZE HEADERSIZE + DATASIZE
#define SLOW_START_DEF 32;
#define seqNumMax 64;

#define SLOW_START_STATE 0
#define CONGESTION_STATE 1
#define FAST_RECOVERY_STATE 2




int sendingData = 1;


struct sendThreadParams
{
    char* hostname;
    unsigned short int port;
    char* filename;
    unsigned long long int bytesToTransfer;
};

void *send_function(void * inparams)
{
    int lastAck = -1;
    int dupeACKcount = 0;
    int ssthresh = SLOW_START_DEF;
    int state = SLOW_START_STATE;


    sendThreadParams * params = (sendThreadParams *) inparams;
    std::ifstream in_file;
    in_file.open(params->filename, std::ifstream::binary);
    std::cout << params->hostname << params->port << params ->filename << params->bytesToTransfer << std::endl;
    if (in_file)
    {
        char port_cstr[6];
        sprintf(port_cstr, "%d", (int) params->port);
        my::sockudp sender;
        int sentBytes = 0;      // number of reliably sent bytes from the file
        char windowSize = 1; 
        char seqNum = 0; // part of TCP header indicates sequence number
        char isEnd = 'F';       // part of TCP header flag indicates last packet        


        in_file.seekg(0, in_file.end);
        int length = in_file.tellg();
        in_file.seekg(0, in_file.beg);

        char packet[PACKETSIZE];
            // Set TCP data
        while(isEnd == 'F'){
            for(int i = 0 ; i < windowSize ; i++)
            {
                in_file.read(packet + HEADERSIZE, DATASIZE);

                // Read less bytes than data size, end of file reached
                if (in_file.gcount() < DATASIZE)
                {
                    isEnd = 'T';
                }

                // Set TCP header
                packet[0] = seqNum;
                packet[1] = isEnd;

                // Send TCP packet
                sender.send(params->hostname, port_cstr, packet, in_file.gcount() + HEADERSIZE);
            }

            char in_ack;
            char ip_buffer[50];
            int rec_bytes = 0;
            my::sockudp rec;
            rec.init_receive(port_cstr);
            int newWindowSize = windowSize;

            std::vector<bool> AcksReceived(windowSize, false);
            int distinctAcksReceived = 0;

            for(int i = 0; i < windowSize ; i++)
            {
                rec_bytes = rec.recOrTimeOut(&in_ack, ip_buffer, 1, 1);
                if (rec_bytes != -1) // if no timeout or error
                {
                    if(AcksReceived[(int)in_ack] == false) // new ACK case
                    {
                        AcksReceived[(int)in_ack] = true;

                        switch(state)
                        {
                            case SLOW_START_STATE:
                                newWindowSize++;
                                dupeACKcount = 0;
                                break;
                            case CONGESTION_STATE:
                                distinctAcksReceived++;
                                break;
                            case FAST_RECOVERY_STATE:
                                newWindowSize = ssthresh;
                                dupeACKcount;
                                state = CONGESTION_STATE;
                                break;
                        }

                    }
                    else if(AcksReceived[(int)in_ack] == true) // dupe
                    {
                        dupeACKcount++;
                        if(dupeACKcount >= 3 && state != FAST_RECOVERY_STATE)
                            break;
                        else if(FAST_RECOVERY_STATE == state)
                        {
                            newWindowSize++;

                        }
                    }
                    std::cout << "Acknolwedged: " << (int)in_ack << std::endl;
                }
                else // TIMEOUT case
                {

                    break;
                }
                if(dupeACKcount >= 3)
                    break;
            }

            windowSize = newWindowSize;

            switch(state)
            {
                case SLOW_START_STATE:
                    if(rec_bytes == -1) // we have timed out
                    {
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = 1;
                        dupeACKcount = 0;

                        //retransmit();
                    }
                    else if(dupeACKcount >= 3)
                    {
                        state = FAST_RECOVERY_STATE;
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = ssthresh + 3;
                    }
                    else if(windowSize >= ssthresh)
                    {
                        state = CONGESTION_STATE;
                    }
                    break;
                case CONGESTION_STATE:
                    if(distinctAcksReceived == windowSize) // all acks received
                    {
                        windowSize++;
                        dupeACKcount = 0;
                    }
                    else if(dupeACKcount >= 3)
                    {
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = ssthresh + 3;

                        state = FAST_RECOVERY_STATE;
                        //retransmit();
                    }
                    else if(rec_bytes == -1) // we have timed out
                    {
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = 1;
                        dupeACKcount = 0;

                        //retransmit();
                    }
                    break;
                case FAST_RECOVERY_STATE:
                    if(rec_bytes == -1) // we have timed out
                    {
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = 1;
                        dupeACKcount = 0;

                        //retransmit();
                    }
                    break;
            }


            seqNum = (seqNum + 1) % 64;
        }
        in_file.close();

    }
   
    pthread_exit(NULL);
}



void *receive_function(void * inport)
{

}

void reliablyTransfer(char* hostname, unsigned short int hostUDPport, char* filename,
                      unsigned long long int bytesToTransfer)
{
    pthread_t send_thread, receive_thread;
    int iret1, iret2;

    // initialize params for sendThread
    sendThreadParams * sendParams = new sendThreadParams;
    sendParams->hostname = hostname;
    sendParams->port = hostUDPport;
    sendParams->filename = filename;
    sendParams->bytesToTransfer = bytesToTransfer;

    // create threads to send data and to receive ACKs
    iret1 = pthread_create(&send_thread, NULL, send_function, (void*) sendParams);
    iret2 = pthread_create(&receive_thread, NULL, receive_function, (void*) &hostUDPport);

    // wait until threads finish
    pthread_join(send_thread, NULL);
    pthread_join(receive_thread, NULL);
    delete sendParams;
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
