#!/bin/bash

#### =========== HELP =================
# Args: 
#   -$1 : output dir -> path to output dir (will be created and will contain models checkpoint / tensorboard / log)
#   -$2 : path to data-bin


#SBATCH -C a100
#SBATCH --time=20:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

OUTPUTDIR=$1
mkdir $OUTPUTDIR

CHECKPOINTDIR=$OUTPUTDIR/checkpoints
mkdir $CHECKPOINTDIR

TENSORBOARD=$OUTPUTDIR/tensorboard
mkdir $TENSORBOARD

LOGFILE="$OUTPUTDIR/train.logs"
echo $LOGFILE

echo $OUTPUTDIR
# Loading modules

source ~/.bashrc

FS=$(which fairseq-hydra-train)

## Data
DATA_DIR=$2 
# Hyperparameters & flags
## Training

set -x

# execution
echo "START TIME: $(date)"
srun $(which fairseq-train) \
  $DATA_DIR \
  --task masked_lm \
  --criterion masked_lm \
  --save-dir $OUTPUTDIR/checkpoint \
  --log-file=$LOGFILE \
  --tensorboard-logdir=$TENSORBOARD \
  --arch masked_lm \
  --encoder-layers 12 \
  --encoder-embed-dim 512 \
  --encoder-attention-heads 8 \
  --encoder-ffn-embed-dim 2048 \
  --dropout 0.1 \
  --tokens-per-sample 512 --skip-invalid-size-inputs-valid-test --batch-size 64 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --adam-eps 1e-06 \
  --lr 0.0005 --lr-scheduler polynomial_decay --warmup-updates 8000 \
  --sample-break-mode complete \
  --max-update 600_000 --update-freq 1 --fp16 --memory-efficient-fp16 \
  --save-interval-updates 5_000 --keep-interval-updates 10 --validate-interval 1000000 --validate-interval-updates 25000 \
  --no-progress-bar --log-format json --log-interval 1000 \
  --total-num-update 600_000 \
  --distributed-backend nccl \
  --distributed-port 16682 \
  --distributed-world-size 128 \
  --ddp-backend no_c10d

echo "END TIME: $(date)"

