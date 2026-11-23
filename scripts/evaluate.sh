#!/bin/bash

models=(../testmodels/M_12b_*B.pt)
../build/play evaluate_multi "${models[@]##*/}" 0.5 48 8 1,2,4 ../ratings/rating_M.txt ../results/test_result_M1.txt