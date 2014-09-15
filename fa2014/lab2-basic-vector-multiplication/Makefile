

##########################################
# INPUT
##########################################
CXX=gcc
DEFINES=-DWB_USE_CUDA
DEFINES+=-DWB_USE_CUSTOM_MALLOC -DWB_USE_COURSERA
CUDA_TOOLKIT_PATH=/usr/local/cuda
CXX_FLAGS=-fPIC -x c++ -O3 -I ./lib/ -I $(CUDA_TOOLKIT_PATH)/include  -static
CXX_FLAGS+=-L $(HOME)/usr/lib -L $(HOME)/usr/cuda/lib -Wall 
CXX_FLAGS+=-I$(HOME)/usr/cuda/include -I$(HOME)/usr/include 
CXX_FLAGS+=-I $(CUDA_TOOLKIT_PATH)/include -L $(CUDA_TOOLKIT_PATH)/lib 
CXX_FLAGS+=$(DEFINES)
LIBS=-arch=sm_21 -lm -lstdc++ -lrt -lcuda -L$(HOME)/usr/lib -L$(CUDA_TOOLKIT_PATH)/lib
ARCH=$(shell uname -s)-x86_64

##########################################
##########################################

SOURCES :=  ./lib/wbArg.cpp        \
			./lib/wbExit.cpp             \
			./lib/wbExport.cpp           \
			./lib/wbFile.cpp             \
			./lib/wbImage.cpp            \
			./lib/wbImport.cpp           \
			./lib/wbInit.cpp             \
			./lib/wbLogger.cpp           \
			./lib/wbMemoryManager.cpp    \
			./lib/wbPPM.cpp              \
			./lib/wbSandbox.cpp          \
			./lib/wbCUDA.cpp			   		 \
			./lib/wbSolution.cpp         \
			./lib/wbTimer.cpp


##############################################
# OUTPUT
##############################################

EXES = sgemm

.SUFFIXES : .o .cpp

OBJECTS = $(SOURCES:.cpp=.o)

##############################################
# OUTPUT
##############################################


.cpp.o:
	$(CXX) $(DEFINES) $(CXX_FLAGS) -c -o $@ $<

$(EXES): $(OBJECTS)
	nvcc $(LIBS) -I ./lib/ -o $@ $(OBJECTS) template.cu

clean:
	-rm -f $(EXES) lib/*.o *~


