../build/simulation/behavior_simulation.d \
 ../build/simulation/behavior_simulation.o: \
 simulation/behavior_simulation.c simulation/behavior_simulation.h
	gcc -g -o ../build/simulation/behavior_simulation.o \
	../build/simulation/behavior_simulation.c
