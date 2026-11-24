make -C ../build
../build/play train MnT_12b_49920.pt MnT_12b_49920.bin 50000 8 1 0.0001 -Ntransfer > ../results/train_result_MnT2.txt 2>&1
