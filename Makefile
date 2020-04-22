exe=main.cu
d=WG
nselect=2
bitmap=1
hash=1
cache=0
bias=1
new: graph.h wtime.h graph.hpp
	nvcc -w -lineinfo -std=c++11  $(exe) -o sampling.bin
		# ./test orkut-ungraph.txt_beg_pos.bin orkut-ungraph.txt_csr.bin
		 ./sampling.bin $d $d/beg.bin $d/csr.bin 1 128 2000 $(nselect) $(bitmap) $(hash) $(cache)
		# May not work for less samples and larger blocks


stream: graph.h wtime.h graph.hpp
	nvcc -w -lineinfo -std=c++11  sampling_gpu4.cu -o stream.bin
		#./test orkut-ungraph.txt_beg_pos.bin orkut-ungraph.txt_csr.bin
		./stream.bin $d/beg.bin $d/csr.bin 100 128 4000

    
