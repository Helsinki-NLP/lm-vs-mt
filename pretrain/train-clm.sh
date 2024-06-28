#!/bin/bash

#### =========== HELP =================
# Args: 
#   -$1 : output dir -> path to output dir (will be created and will contain models checkpoint / tensorboard / log)
#   -$2 : path to data-bin


#SBATCH -C a100
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#cd ${SLURM_SUBMIT_DIR}

# Set directories
OUTPUTDIR=$1
mkdir $OUTPUTDIR

CHECKPOINTDIR=$OUTPUTDIR/checkpoints
mkdir $CHECKPOINTDIR

TENSORBOARD=$OUTPUTDIR/tensorboard
mkdir $TENSORBOARD

LOGFILE="$OUTPUTDIR/train.logs"
cp $2 $OUTPUTDIR/

echo $OUTPUTDIR
echo $CONFIGDIR
echo $CONFIGNAME


# Loading modules

## Data
DATA_DIR=$2 
# Hyperparameters & flags
## Training

set -x

# execution
echo "START TIME: $(date)"
srun $(which fairseq-train) \
  $DATA_DIR \
  --task language_modeling \
  --save-dir $OUTPUTDIR/checkpoint \
  --log-file=$LOGFILE \
  --tensorboard-logdir=$TENSORBOARD \
  --arch transformer_lm \
  --decoder-layers 12 \
  --decoder-embed-dim 512 \
  --decoder-attention-heads 8 \
  --decoder-ffn-embed-dim 2048 \
  --dropout 0.1 \
  --tokens-per-sample 512 --skip-invalid-size-inputs-valid-test --batch-size 32 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --adam-eps 1e-06 \
  --lr 0.0005 --lr-scheduler polynomial_decay --warmup-updates 8000 \
  --sample-break-mode complete \
  --max-update 600_000 --update-freq 1 --fp16 --memory-efficient-fp16 \
  --save-interval-updates 5_000 --keep-interval-updates 10 --validate-interval 0 --validate-interval-updates 25000 \
  --no-progress-bar --log-format json --log-interval 1000 \
  --total-num-update 600_000 \
  --distributed-backend nccl \
  --distributed-port 16682 \
  --distributed-world-size 8 \
  --ddp-backend no_c10d

echo "END TIME: $(date)"

