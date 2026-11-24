#!/bin/bash

abmodels=(../testmodels/MnT_12b_*B.pt)
models=(../testmodels/M_12b_*.pt)

for i in "${!abmodels[@]}"; do
    abmodel="${abmodels[$i]}"
    model="${models[$i]}"
    ../build/play evaluate_two "${model[@]##*/}" "${abmodel[@]##*/}" 0.5 48 8
done

# ../build/play evaluate_multi "${models[@]##*/}" 0.5 48 8 1,2,4 ../ratings/rating_MnT.txt ../results/test_result_MnS.txt