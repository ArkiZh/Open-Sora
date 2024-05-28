set -x
OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=true NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc-per-node=8 --rdzv-id=sora --rdzv-backend=c10d --rdzv-endpoint=ks-gpu-2:19901 train.py config.py --data-path /slurmhome/kzhang/datasets/HD-VG-130M/data.csv --load /slurmhome/kzhang/repos/Open-Sora/run/outputs/STDiT2-XL-2-001/epoch47-global_step43700
