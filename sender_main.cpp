#include "socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <queue>
#include <vector>

#define DATASIZE 1450
#define HEADERSIZE 5
#define PACKETSIZE HEADERSIZE + DATASIZE
#define SLOW_START_DEF 64;

#define SLOW_START_STATE 0
#define CONGESTION_STATE 1
#define FAST_RECOVERY_STATE 2


void reliablyTransfer(char* hostname, unsigned short int hostUDPport, char* filename,
                      unsigned long long int bytesToTransfer)
{
    int lastAck = -1;
    int dupeACKcount = 0;
    int ssthresh = SLOW_START_DEF;
    int state = SLOW_START_STATE;

    //unsigned long long bytesLeft = bytesToTransfer;

    std::ifstream in_file;
    in_file.open(filename, std::ifstream::binary);
    if (in_file)
    {
        char port_cstr[6];
        sprintf(port_cstr, "%d", (int) hostUDPport);
        my::sockudp sender;

        int sentBytes = 0;      // number of reliably sent bytes from the file
        int windowSize = 1; 
        int seqNum = 0;         // part of TCP header indicates sequence number
        char isEnd = 'F';       // part of TCP header flag indicates last packet        


        in_file.seekg(0, in_file.end);
        int length = in_file.tellg();
        in_file.seekg(0, in_file.beg);

        char packet[PACKETSIZE];

        int in_ack;
        char ip_buffer[50];
        int rec_bytes = 0;
        my::sockudp rec;
        rec.init_receive(port_cstr);
        int newWindowSize = windowSize;

        bool retransmit = false;
        int beginPos = in_file.tellg();
        int beginSeq = 0;
        int lastPacket = 0x0FFFFFFF;

            // Set TCP data
        while(seqNum < lastPacket)
        {
            retransmit = false;
            beginPos = in_file.tellg();
            beginSeq = seqNum;
            
            for(int i = 0 ; i < windowSize ; i++)
            {
                if(seqNum > lastPacket)
                    break;

                unsigned long long int toRead = DATASIZE;
                if((unsigned long long)(seqNum + 1) * DATASIZE > bytesToTransfer) // if next packet to read, is too large, clamp it
                    toRead = bytesToTransfer - ( seqNum * DATASIZE );
                in_file.read(packet + HEADERSIZE, toRead);

                // Read less bytes than data size or end of file reached
                if ( (in_file.gcount() < DATASIZE) || in_file.eof() )
                {
                    isEnd = 'T';
                    windowSize = i+1;
                    #ifdef DEBUG
                    std::cout << "Reached EOF\n";
                    #endif
                    lastPacket = seqNum;
                    in_file.clear();
                }

                // Set TCP header
                memcpy(packet, &seqNum, 4);
                packet[4] = isEnd;

                // Send TCP packet
                #ifdef DEBUG
                std::cout << "SEND: " << *((int *) packet) << std::endl;
                #endif
                sender.send(hostname, port_cstr, packet, in_file.gcount() + HEADERSIZE);
                seqNum++;
                if(isEnd == 'T')
                    break;

            }
            #ifdef DEBUG
	    std::cout << "WindowSize: " << windowSize << std::endl;
        #endif
            std::vector<bool> AcksReceived(windowSize, false);
            int distinctAcksReceived = 0;
            #ifdef DEBUG
    	    std::cout << "State: " << state << std::endl;
            #endif
    	    newWindowSize = windowSize;
            for(int i = 0; i < windowSize ; i++)
            {
                rec_bytes = rec.recOrTimeOut((char*) &in_ack, ip_buffer, 100, 4);
                if (rec_bytes != -1) // if no timeout or error
                {
                    #ifdef DEBUG
                    std::cout << "Begin: " << beginSeq << " Received ACK: " << in_ack << std::endl;
                    #endif
                    int this_ack = in_ack;

                    in_ack = in_ack - beginSeq;
                    if(in_ack < 0)
                    {
                            dupeACKcount++;
                            if(dupeACKcount >= 3)
                                if(state == SLOW_START_STATE)
                                    break;
                                else if (state == CONGESTION_STATE)
                                {
                                    ssthresh = windowSize/2;
                                    newWindowSize += 3;
                                    state = FAST_RECOVERY_STATE;
                                    break;
                                }
                            else if(FAST_RECOVERY_STATE == state)
                                newWindowSize++;
                        continue;
                    }

                    //if(AcksReceived[in_ack] == false) // new ACK case  tempZ
                    else if(this_ack > lastAck) // tempZ
                    {
                        lastAck = this_ack; // tempZ
                        //AcksReceived[in_ack] = true; // tempZ


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
                                dupeACKcount = 0;
                                state = CONGESTION_STATE;
                                break;
                        }
                    }
                    //else if(AcksReceived[in_ack] == true) // dupe tempZ
                    else if(this_ack <= lastAck) // tempZ
                    {

                        dupeACKcount++;

                        if(dupeACKcount >= 3 && state != FAST_RECOVERY_STATE)
                            break;
                        else if(FAST_RECOVERY_STATE == state)
                            newWindowSize++;
                    }
                }
                else // TIMEOUT case
                {
                    #ifdef DEBUG
                    std::cout << "timeout occurred" << std::endl;
                    #endif
                    break;
                }
                if(dupeACKcount >= 3)
                {
                    // clear out udp buffer
                    while(1)
                    {

                        int in_ack = -1;
                        rec_bytes = rec.recOrTimeOut((char*) &in_ack, ip_buffer, 100, 4);
                        if(rec_bytes == -1) // on timeout
                            break;
                        else
                        {
                            newWindowSize++;
                            #ifdef DEBUG
                            std::cout<<"processed dupe ack" << std::endl;
                            #endif
                        }

                    }
                    break;
                }
                    
            }

            windowSize = newWindowSize;
            #ifdef DEBUG
     	    std::cout<< "windowSize: " << windowSize << std::endl;
            std::cout<< "distinctACKs: " << distinctAcksReceived << std::endl;
            std::cout<< "lastAck: " << lastAck << std::endl;
            #endif
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

                        retransmit = true;
                    }
                    else if(dupeACKcount >= 3)
                    {
                        state = FAST_RECOVERY_STATE;
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = ssthresh + 3;
                        retransmit = true;
                    }
                    else if(windowSize >= ssthresh)
                    {
                        state = CONGESTION_STATE;
                    }
                    break;
                case CONGESTION_STATE:
                    if(rec_bytes == -1) // we have timed out
                    {
                        ssthresh = windowSize/2;
                        if(ssthresh < 1)
                            ssthresh = 1;
                        windowSize = 1;
                        dupeACKcount = 0;
			state = SLOW_START_STATE;
                        retransmit = true;
                    }
                    //else if(distinctAcksReceived == windowSize) // all acks received tempZ
                    else if(lastAck == windowSize + beginSeq - 1)
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
                        retransmit = true;
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
                        state = SLOW_START_STATE;
                        retransmit = true;
                    }
                    break;
            }
            #ifdef DEBUG
            std::cout << "Retransmit: " << retransmit << std::endl;
            #endif
            if(retransmit)
            {
                //for(int i=0; i < AcksReceived.size() ; i++) // tempZ
                {
//                  if(!AcksReceived[i])
                    {
//                        in_file.seekg(beginPos + DATASIZE * i, in_file.beg); // tempZ
                        in_file.seekg(DATASIZE * (lastAck + 1), in_file.beg); // tempZ
                        //seqNum = beginSeq + i; // tempZ
                        

                        //break; // tempZ
                    }
                }
            }
            
            in_file.seekg(DATASIZE * (lastAck + 1), in_file.beg); // tempZ
            seqNum = lastAck+1; // tempZ

        }
        in_file.close();
    }
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
