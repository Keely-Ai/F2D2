export NCCL_P2P_DISABLE=1
torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$2 train.py  \
    --outdir=<OUTDIR> --data=<DATA_PATH> --precond=sc \
    --cond=0 --arch=ddpmpp --metrics=fid50k_full    \
    --duration=300 --tick=50 --batch=512 --lr=0.0001 --dropout=0.13 --augment=0.0  \
    --wandb=True --wandb_key='<WANDB_KEY>'  \
    --optim=torch.optim.RAdam \
    --sample_every=100 --eval_every=100 \
    --teacher=<PATH_TO_SHORTCUT_DISTILL_MODEL>  \
    --resume=<PATH_TO_SHORTCUT_DISTILL_MODEL>  \
    ${@:3}