set -x 

read -r -d '' training_commands <<EOF
examples/train_pcl.py \
     --save_path /workspace/zihan/CODE/OpenRLHF/checkpoint/qwen-coder-7b-rl-0102 \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 8 \
     --micro_train_batch_size 1 \
     --pretrain /workspace/zihan/MODEL/base_model/qwen-coder-7b-instruct/ \
     --bf16 \
     --lora_rank 64 \
     --lora_alpha 64 \
     --max_epochs 2 \
     --max_len 16000 \
     --zero_stage 3 \
     --beta 0.03 \
     --learning_rate 1e-7 \
     --critic_learning_rate 5e-5 \
     --adam_offload \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload \
     --dro_actor_loss \
     --rew_mul 1 \
     --rew_add 0 \
     --kl_reg 0.01 \
     --unbiased_kl \
     --plot_weights \
     --padding_side left \
     --packing_samples \
     --only_critic_lora \
     --ring_attn_size 4 \
     --ring_head_stride 1 \
     --train_file /workspace/zihan/CODE/mle-syn/save_data/12-31/OREO/MLInstruct_rel_rl.jsonl \
     --use_wandb 3d0e38b972f4b849c7e75d05b4d3d86b861de97f
EOF
     # --use_wandb 3d0e38b972f4b849c7e75d05b4d3d86b861de97f
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]

# INCLUDE=localhost:$CUDA_VISIBLE_DEVICES
# unset CUDA_VISIBLE_DEVICES
if [[ ${1} != "slurm" ]]; then
     deepspeed $training_commands
fi
