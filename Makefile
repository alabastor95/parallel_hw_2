# System dependent definitions

# For ARCHER
CC=	c++
CPPFLAGS=	-O3 -g -fopenmp
LFLAGS= -fopenmp

# For Cirrus or any standard Linux system
#CC=	icc
#CPPFLAGS=	-O3
#LFLAGS=	-lm


# System independent definitions

MF=	Makefile

EXE=	cfd

INC= \
	boundary.h \
	cfdio.h \
	jacobi.h

SRC= \
	boundary.cpp \
	cfd.cpp \
	cfdio.cpp \
	jacobi.cpp

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .cpp .o

OBJ=	$(SRC:.cpp=.o)



all:	$(EXE)

$(OBJ):	$(INC)

$(EXE):	$(OBJ)
	$(CC) $(CPPFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) velocity.dat colourmap.dat cfd.plt core out.png

.cpp.o:
	$(CC) $(CPPFLAGS) -c $<

