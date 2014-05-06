TARGETS= receiver_main sender_main

CC=g++
CFLAGS=-Wall -g -Werror -std=c++0x -Wextra
PTHREAD=-lpthread -lm

.PHONY: all clean

all: $(TARGETS)

receiver_main: receiver_main.o socket.o
	$(CC) $(CFLAGS) receiver_main.o socket.o -o reliable_receiver $(PTHREAD)

sender_main: sender_main.o socket.o
	$(CC) $(CFLAGS) sender_main.o socket.o -o reliable_sender $(PTHREAD)

%.o: %.cpp %.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o *.exe
	rm -f reliable_sender reliable_receiver

