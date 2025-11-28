input=$1
output=$2

if [ -z "$output" ]; then
    ./train < "$input"
elif [ "$output" = "none" ]; then
    ./train < "$input" > /dev/null 2>&1
else
    ./train < "$input" > "$output"
fi
