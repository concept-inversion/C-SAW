exe =streaming.bin
commflags= --compiler-options -Wall -Xptxas -v -Xcudafe -\# --resource-usage 
#commflags+=--generate-line-info --source-in-ptx 
#commflags+=-dryrun
commflags+= -std=c++11
#commflags+=-keep -keep-dir ptx
#commflags+=-maxrregcount=64
cucc= "$(shell which nvcc)"
cc= "$(shell which gcc)"
cuflags= -arch=sm_35  $(commflags) #-Xptxas -dlcm=cg#disable l1 cache

#ifeq ($(monitor), 1)
#	cuflags+= -DENABLE_MONITORING
#endif
#
#ifeq ($(check), 1)
#	cuflags+= -DENABLE_CHECKING
#endif

ifeq ($(debug), 1)
	cuflags+= -G -g -O0
	cflags += -g -O0
else
	cflags += -O3
	cuflags+= -O3
endif


#cuflags+= -ccbin=g++ -Xcompiler -fopenmp

objs	= $(patsubst %.cu,%.o,$(wildcard *.cu))

deps	= $(wildcard ./*.h) \
		  $(wildcard ./*.cuh) \
		  $(wildcard ./*.hpp) \
		  $(wildcard ./*.h) \
		  Makefile

%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $(ldflags) $< -o $@

$(exe):$(objs)
	$(cucc) $(objs) $(cuflags) -o $(exe)	

clean:
	rm -rf *.o ${exe}
