echo "FF"

echo " ITS Re-"
for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
    ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 1 1 1 0
done

echo "select-Baseline"
for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
    ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 0 1 1 0      
done


echo " Normalize"
for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
    ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 2 1 1 0
done

# echo "Normalize + bitmap"
# for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
#     ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 1 1 0 0 0   
# done

# echo "hash"
# for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
#     ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 1 1 1 0     
# done

# echo "hash+cache"
# for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
#     ./sampling.bin $d $d/beg.bin $d/csr.bin 100 32 2000 1 1 1 1     
# done

#  echo "combined"
# for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
# echo $d
#     ./baseline.bin $d/beg.bin $d/csr.bin 100 32 2000   
# done

# echo "baseline"
# echo $d
# for d in /gpfs/alpine/proj-shared/csc289/Sampling/*/ ; do
#     ./combined.bin $d/beg.bin $d/csr.bin 100 32 2000    
# done