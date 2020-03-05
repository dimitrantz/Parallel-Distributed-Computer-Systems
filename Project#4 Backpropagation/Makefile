# Makefile for the Project in Parallel Programming, 2016-2017

CFLAGS = -I $(INCDIR)

SRCDIR = src/
INCDIR = include/
BLDDIR = build/
BINDIR = bin/
DEPS = $(INCDIR)utils.h
SRCS = main.c functions.c backpropagation.c feedforward.c outputerrorcalculation.c updateweights.c 
OBJS = $(patsubst %.c,$(BLDDIR)%.o,$(SRCS))
EXE  = $(BINDIR)backpropagation

$(BLDDIR)%.o: $(SRCDIR)%.c $(DEPS)
	nvcc -x cu -arch=sm_20 $(CFLAGS). -dc $< -o $@
	
$(EXE): $(OBJS)
	nvcc -arch=sm_20 $(CFLAGS) -o $@ $^ -lm

clean:
	$(RM) $(BLDDIR)*.o $(EXE)
	
