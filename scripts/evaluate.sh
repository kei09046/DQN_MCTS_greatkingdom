#!/usr/bin/env bash
set -euo pipefail  # safer bash settings

# Arrays of files
abmodels=(../testmodels/MnTnP_12b_*B.pt)
models=(../testmodels/M_12b_*.pt)

for abmodel in "${abmodels[@]}"; do
    # Extract ID part from abmodel filename
    # Example: MnTnP_12b_ABC123B.pt → ABC123
    abfile=$(basename "$abmodel")
    idpart=$(echo "$abfile" | sed -E 's/^MnTnP_12b_(.*)B\.pt$/\1/')

    # Find matching model file
    match=""
    for model in "${models[@]}"; do
        modfile=$(basename "$model")
        if [[ "$modfile" == "M_12b_${idpart}.pt" ]]; then
            match="$model"
            break
        fi
    done

    # Run evaluation if match found
    if [[ -n "$match" ]]; then
        echo "Evaluating: $match vs $abmodel"
        ../build/play evaluate_two "$(basename "$match")" "$(basename "$abmodel")" 0.5 48 8
    else
        echo "No matching model found for $abfile" >&2
    fi
done


# ../build/play evaluate_multi "${models[@]##*/}" 0.5 48 8 1,2,4 ../ratings/rating_MnT.txt ../results/test_result_MnS.txt