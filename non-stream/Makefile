exe=streaming.bin
N=1
cucc= "$(shell which nvcc)"
cc= "$(shell which mpicxx)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64
cuflags= --compiler-options -v -Xcudafe  -\# --resource-usage 
cuflags+= -std=c++11
objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \
	$(wildcard ./*.hpp) \
	$(wildcard ./*.h) \


%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $< -o $@

%.o:%.cpp $(deps)
	$(cc) -c  $< -o $@

$(exe):$(objs)
	$(cc) $(objs) $(commflags) -O3 -o $(exe)


test:$(exe)	
	#Multidimensional random walk
	mpirun -n $(N) $(exe) LJ LJ/beg.bin LJ/csr.bin 1 32 1 20 1 10 1
	
clean:
	rm -rf *.o ${exe}
