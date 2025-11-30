input=$1
output=$2

if [ -z "$output" ]; then
    ./train < "$input"
elif [ "$output" = "none" ]; then
    ./train < "$input" > /dev/null 2>&1
else
    ./train < "$input" > "$output"
fi

# usage : ./train run_train.txt
# example run_train.txt :
# train
# modelg2000.pt (model name to start at, to start from beginning, type none)
# 10000 4 0 (total games to train, number of threads, is_shown)