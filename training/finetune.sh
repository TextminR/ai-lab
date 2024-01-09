#!/bin/sh

# == USER SETTINGS == #
HF_TOKEN=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
WANDB_TOKEN=8949230cd2fb227b199598ff49dcd45107ae63aa
AUTOSTOP=0

# check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_TOKEN" ];
then
    echo "Gotcha! Please set HF_TOKEN and WANDB_TOKEN before continuing."
    exit 1
fi
export HF_TOKEN=$HF_TOKEN

# login to wandb
echo "Logging in to wandb..."
wandb login $WANDB_TOKEN
if [ $? -ne 0 ]; then
    echo "Failed to login to wandb."
    exit 1
fi

echo "Everything is set up. Starting training..."
python ./scripts/sft.py $@

# stop pod when training is finished
if [ $AUTOSTOP -eq 1 ]; then
    echo "Shutting down..."
    runpodctl stop pod $RUNPOD_POD_ID
fi