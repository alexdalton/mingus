TARGETS= receiver_main sender_main

CC=g++
CFLAGS=-Wall -g -Werror -std=c++0x -Wextra

.PHONY: all clean

all: $(TARGETS)

receiver_main: receiver_main.o socket.o
	$(CC) $(CFLAGS) receiver_main.o socket.o -o reliable_receiver

sender_main: sender_main.o socket.o
	$(CC) $(CFLAGS) sender_main.o socket.o -o reliable_sender

%.o: %.cpp %.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o *.exe


