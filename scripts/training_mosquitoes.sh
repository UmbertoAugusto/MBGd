#!/bin/bash

cat << "EOF"
  __  __ ____   _____ _____  
 |  \/  |  _ \ / ____|  __ \ 
 | \  / | |_) | |  __| |  | |
 | |\/| |  _ <| | |_ | |  | |
 | |  | | |_) | |__| | |__| |
 |_|  |_|____/ \_____|_____/ 
                             
                             
EOF
echo " --- Welcome to MBGd workflow ! ---"

# Load configurations from config.yaml
CONFIG_FILE="configs/config.yaml"

# Path to Local yq (linux)
YQ_LOCAL="./yq"

# Verifica se o yq local existe
if [ ! -f "$YQ_LOCAL" ]; then
    echo "yq not found. Downloading yq..."
    wget https://github.com/mikefarah/yq/releases/download/v4.35.2/yq_linux_amd64 -O yq
    chmod +x yq
fi

# Extract values from config.yaml usando yq local
OBJ=$($YQ_LOCAL e '.TRAINING.OBJECT' $CONFIG_FILE)
FOLDS=$($YQ_LOCAL e '.REGISTER_DATASETS.FOLDS' $CONFIG_FILE)
DATATYPE=$($YQ_LOCAL e '.REGISTER_DATASETS.DATATYPE' $CONFIG_FILE)
MAX_ITER=$($YQ_LOCAL e '.TRAINING.MAX_ITER' $CONFIG_FILE)
CUDA_DEVICE=$($YQ_LOCAL e '.TRAINING.CUDA_DEVICE // 0' $CONFIG_FILE)

# START MOSQUITOES WORKFLOW
echo "Starting training workflow for ${OBJ} detection, ${FOLDS} folds."

for ((outer_fold=1; outer_fold<=FOLDS; outer_fold++)); do 

    for ((inner_fold=1; inner_fold<=5; inner_fold++)); do
        if (( inner_fold == outer_fold )); then
            continue
        fi

        echo ""
        echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
        duration=$SECONDS
        echo "Time spent: $((duration / 3600)) hours, $(((duration / 60) % 60)) minutes and $((duration % 60)) seconds"
        echo ""

        echo "Running train.py with fold $inner_fold for validation and fold $outer_fold for test..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/train.py --config-file $CONFIG_FILE --test-fold "$outer_fold" --val-fold "$inner_fold" --object "$OBJ"
    
        echo "Running validation.py with fold $inner_fold for validation and fold $outer_fold for test..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/validation.py --config-file $CONFIG_FILE --test-fold "$outer_fold" --val-fold "$inner_fold" --object "$OBJ" --datatype "$DATATYPE"

        echo "Running test.py with fold $inner_fold for validation and fold $outer_fold for test..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/test.py --config-file $CONFIG_FILE --test-fold "$outer_fold" --val-fold "$inner_fold" --object "$OBJ" --datatype "$DATATYPE"

    done
done

# FINISH MOSQUITOES WORKFLOW
echo ""
echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
duration=$SECONDS
echo "Time spent: $((duration / 3600)) hours, $(((duration / 60) % 60)) minutes and $((duration % 60)) seconds"
echo ""

echo "Training workflow completed."