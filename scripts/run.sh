make -C ../build
../build/play train M_12b_253440.pt M_12b_253440.bin 1000000 8 1 0.00001 >> ../results/train_result_M5.txt 2>&1
