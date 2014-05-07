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
    int count = 0;
    unsigned long long bytesLeft = bytesToTransfer;

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
        int beginSeq = 0;

        // Set TCP data
        while(isEnd == 'F' || retransmit)
	{
            retransmit = false;
            beginSeq = seqNum;
            
            for(int i = 0 ; i < windowSize ; i++)
            {
                in_file.seekg(DATASIZE * seqNum, in_file.beg);
		unsigned long long position = in_file.tellg();
		if((position + DATASIZE) >=  bytesToTransfer)
		{
			in_file.read(packet + HEADERSIZE, bytesToTransfer - in_file.tellg());
			isEnd = 'T';
			windowSize = i + 1;
		}
		else
		{
                	in_file.read(packet + HEADERSIZE, DATASIZE);
		}

                // Read less bytes than data size or end of file reached
                if ( (in_file.gcount() < DATASIZE) || in_file.eof())
                {
                    isEnd = 'T';
                    windowSize = i+1;
		    //std::cout << "Reached EOF\n";
                }

                // Set TCP header
                memcpy(packet, &seqNum, 4);
                packet[4] = isEnd;

                // Send TCP packet
                //std::cout << i + 1 << " of " << windowSize<< " SEND: " << *((int *) packet) << std::endl;

		count++;
		//if(count % 20)
                    sender.send(hostname, port_cstr, packet, in_file.gcount() + HEADERSIZE);
                seqNum++;
                if(isEnd == 'T')
                    break;

            }
            
	    //std::cout << "WindowSize: " << windowSize << std::endl;
            std::vector<bool> AcksReceived(windowSize, false);
            int distinctAcksReceived = 0;
	    //std::cout << "State: " << state << std::endl;
	    newWindowSize = windowSize;
            for(int i = 0; i < windowSize ; i++)
            {
                rec_bytes = rec.recOrTimeOut((char*) &in_ack, ip_buffer, 10, 4);
                if (rec_bytes != -1) // if no timeout or error
                {
		    //std::cout << i + 1 << " of " << windowSize << " Received ACK: " << in_ack << std::endl;
                    in_ack = in_ack - beginSeq;
		    if(in_ack < 0)
		    {
                        dupeACKcount++;
                        if(dupeACKcount >= 3 && state != FAST_RECOVERY_STATE)
                            break;
                        else if(FAST_RECOVERY_STATE == state)
                            newWindowSize++;
			continue;
 		    }
                    if(AcksReceived[in_ack] == false) // new ACK case
                    {

                        AcksReceived[in_ack] = true;
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
                    else if(AcksReceived[in_ack] == true) // dupe
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
                    break;
                }
                if(dupeACKcount >= 3)
                    break;
            }

            windowSize = newWindowSize;
 	    //std::cout<< "windowSize: " << windowSize << std::endl;
	    //std::cout<< "distinctACKs: " << distinctAcksReceived << std::endl;
	    int nextSeqNum;
	    bool useNext = false;
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
			while(1)
			{
			    int extra_ack;
			    int extra_bytes;
                	    extra_bytes = rec.recOrTimeOut((char*) &extra_ack, ip_buffer, 1, 4);

			    if(extra_bytes == -1)
			    {
 				break;
			    }
			    int index = extra_ack - beginSeq;
			    if (index < 0)
			    {
			    	windowSize++;
			        nextSeqNum = extra_ack + 1;
				useNext = true;
				continue;
			    }
			    else if(AcksReceived[(extra_ack - beginSeq)] == true)
			    {
			        windowSize++;
			    }
			    //std::cout << "test1" << std::endl; 
			}

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
                    else if(distinctAcksReceived == windowSize) // all acks received
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
			while(1)
			{
			    int extra_ack;
			    int extra_bytes;
                	    extra_bytes = rec.recOrTimeOut((char*) &extra_ack, ip_buffer, 1, 4);
			    if(extra_bytes == -1)
			    {
 				break;
			    }
			    int index = extra_ack - beginSeq;
			    if (index < 0)
			    {
			    	windowSize++;
			        nextSeqNum = extra_ack + 1;
				useNext = true;
				continue;
			    }
			    else if(AcksReceived[(extra_ack - beginSeq)] == true)
			    {
			        windowSize++;
			    }
			    //std::cout << "test1" << std::endl; 
			}
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
            //std::cout << "Retransmit: " << retransmit << std::endl;
            if(retransmit)
            {
		if(useNext)
		{
		    useNext = false;
		    seqNum = nextSeqNum;
		}
		else
		{
                    for(int i=0; i < AcksReceived.size() ; i++)
                    {
                        if(AcksReceived[i])
                        {
                            seqNum = beginSeq + i + 1;
                            //break;
                        }
                    }
		} 
            }
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
