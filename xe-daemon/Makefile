CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lze_loader -lm -lpthread

xe-daemon: xe-daemon.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: xe-daemon
	install -m 755 xe-daemon /usr/local/bin/

clean:
	rm -f xe-daemon

.PHONY: install clean
