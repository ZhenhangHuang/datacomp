set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
MASTER_ADDR="0.0.0.0"
MASTER_PORT=12802
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# DATA_PATH="/mnt/petrelfs/huangzhenhang/datasets/datacomp/shards"
DATA_PATH="./out/laion2b/"
SCALE="small"
SEED=0
OUTPUT_DIR="./out/"
NUM_CHECKPOINTS=8
EXP_NAME="datacomp-scale-${SCALE}-seed${SEED}_basic-filter"
PRECISION="amp"  # You can also use amp_bfloat16 if supported by your hardware.

# Change comment as needed
srun \
    --partition=${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    ${SRUN_ARGS} \
    python train.py \
    --scale ${SCALE} \
    --data_dir ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --precision ${PRECISION} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --seed ${SEED} \
    --report_to_wandb \
    --dataset_resampled \
    --accum_freq 1 ${@:4}
    
    # --cpu_bind=v --accel-bind=gn \
