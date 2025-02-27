# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -std=c99

# Source files
SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)

# Executable name
EXECUTABLE = quantum_cube

# Default target
all: $(EXECUTABLE)

# Rule to compile the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

# Rule to compile object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Phony targets
.PHONY: all clean
