
exp_name=bart-lm-600k
mkdir -p $exp_name log 

data_dir=/path/to/pretrain/bart-lm/data_bin

srun $(which fairseq-train) $data_dir \
  --save-dir ./$exp_name \
  --arch bart_base --layernorm-embedding \
  --task denoising \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.1 \
  --batch-size 64 --max-tokens 8192 --update-freq 1 \
  --max-update 600_000 --fp16 \
  --save-interval 1 --save-interval-updates 5_000 --keep-interval-updates 10 \
  --no-progress-bar --log-format json --log-interval 5_000 \
  --log-file ./log/$exp_name-$SLURM_JOB_ID.log \
  --distributed-backend nccl \
  --distributed-port 16672 \
  --distributed-world-size $SLURM_NTASKS \
  --distributed-num-procs $SLURM_NTASKS \
  --ddp-backend pytorch_ddp