set -x 
read -r -d '' training_commands <<EOF
examples/train_pcl.py \
     --save_path /workspace/haoran/models/test \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 8 \
     --micro_train_batch_size 1 \
     --pretrain /workspace/haoran/models/Qwen/Qwen2.5-Coder-1.5B-Instruct/ \
     --bf16 \
     --max_epochs 1 \
     --max_len 32768 \
     --zero_stage 3 \
     --beta 0.03 \
     --learning_rate 5e-6 \
     --critic_learning_rate 5e-6 \
     --adam_offload \
     --flash_attn \
     --gradient_checkpointing \
     --dro_actor_loss \
     --step_level \
     --kl_reg 0.01 \
     --unbiased_kl \
     --ref_offload \
     --plot_weights \
     --lora_rank 64 \
     --lora_alpha 64 \
     --padding_side left \
     --packing_samples \
     --ring_attn_size 2 \
     --ring_head_stride 2 \
     --task alfworld \
     --only_critic_lora \
     --train_file /workspace/haoran/datasets/codeagt.json
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
    deepspeed --include $INCLUDE $training_commands
fi

# # #! /bin/bash
# # set -o pipefail

# # source $1
# # mkdir -p logs/${MLP_TASK_ID}
# # mpi_cmd="mpirun -np ${MLP_WORKER_NUM} \
# #         --hostfile ${MLP_MPI_HOSTFILE} \
# #         -npernode 1 -x OMPI_MCA_btl_tcp_if_include=$MLP_SOCKET_IFNAME \
# #         pip install datasets deepspeed einops jsonlines ray[default] tqdm transformers accelerate wandb tiktoken mpi4py"
# # eval ${mpi_cmd}
# # read -r -d '' training_commands <<EOF
# # examples/train_pcl.py \
# #      --save_path /workspace/haoran/models/Qwen/Qwen2.5-Coder-1.5B-Instruct/ \
# #      --save_steps -1 \
# #      --logging_steps 1 \
# #      --eval_steps -1 \
# #      --train_batch_size 8 \
# #      --micro_train_batch_size 1\
# #      --pretrain /workspace/haoran/models/test \
# #      --bf16 \
# #      --max_epochs 1 \
# #      --max_len 1024 \
# #      --zero_stage 3 \
# #      --beta 0.03 \
# #      --learning_rate  5e-6 \
# #      --critic_learning_rate 5e-6 \
# #      --adam_offload \
# #      --flash_attn \
# #      --gradient_checkpointing \
# #      --dro_actor_loss \
# #      --kl_reg 0.01 \
# #      --unbiased_kl \
# #      --ref_offload \
# #      --plot_weights \
# #      --lora_rank 64 \
# #      --lora_alpha 64 \
# #      --padding_side left \
# #      --packing_samples \
# #      --ring_attn_size 8 \
# #      --ring_head_stride 1 \
# #      --task alfworld \
# #      --only_critic_lora \
# #      --train_file /workspace/haoran/datasets/codeagt.json \
# #      --use_mpi_init
# # EOF

# # if [[ -z "$NCCL_IB_QPS_PER_CONNECTION" ]]; then
# #   NCCL_IB_QPS_PER_CONNECTION=2
# # fi
# # echo "Start training"
# # mpi_cmd="mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
# #         --hostfile ${MLP_MPI_HOSTFILE} \
# #         --allow-run-as-root -oversubscribe -map-by ppr:8:node \
# #         -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=$MLP_SOCKET_IFNAME \
# #         --output-filename logs/${MLP_TASK_ID} \
# #         -x NCCL_PXN_DISABLE=0 \
# #         -x NCCL_IB_GID_INDEX=3 \
# #         -x NCCL_NET_GDR_LEVEL=4 \
# #         -x NCCL_IB_RETRY_CNT=7 \
# #         -x NCCL_IB_TIMEOUT=50 \
# #         -x NCCL_IB_QPS_PER_CONNECTION=$NCCL_IB_QPS_PER_CONNECTION \
# #         -x NCCL_P2P_LEVEL=NVL \
# #         -x NCCL_DEBUG=VERSION \
# #         -x NCCL_IB_TC=106 \
# #         -x PATH \
# #         -x MASTER_ADDR=$MLP_WORKER_0_HOST \
# #         -x MASTER_PORT=$MLP_WORKER_0_PORT \
# #         -x GLOO_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
# #         -x NCCL_SOCKET_IFNAME=$MLP_SOCKET_IFNAME \
# #         -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
# #         -x NCCL_NVLS_ENABLE=0 \
# #         -x NVTE_FWD_LAYERNORM_SM_MARGIN=8 \
# #         -x NVTE_BWD_LAYERNORM_SM_MARGIN=8 \
# #         -x LD_LIBRARY_PATH=/workspace/yao/nccl/build/lib:$LD_LIBRARY_PATH \
# #         python ${training_commands} --use_mpi_init"
# # echo ${mpi_cmd}
# # eval ${mpi_cmd} 2>&1 | tee logs/${MLP_TASK_ID}/output.log