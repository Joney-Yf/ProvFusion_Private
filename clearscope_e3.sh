#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate orthrus

# CLEARSCOPE_E3 comprehensive sweep (dim=64, seed=1).
# Baseline config (v3): dim_64 nhd_4 nh_0.1 mp_200 mpf_50 wd_0.001 wdf_0.0001 -> TP=6 FP=7
# Array order puts the baseline neighborhood first.
learning_rates=(0.0015)
lr_fs=(0.001)
mask_rates=(0.1 0.3)
layers=(2)
seeds=(1)
epochs=(200 50 5)
num_hiddens=(64)
replace_rates=(0.0)
weight_decays=(0.001 0.01 0.0001 1e-5 1e-6 0)
num_heads=(4 2)
max_epoch_fs=(50 100 200)
weight_decay_fs=(0.0001 5e-5 1e-5 2e-6 1e-6 0)
drop_edge_rates=(0.0)
in_drops=(0.2)
attn_drops=(0.1)
alpha_ls=(3)
dataset="CLEARSCOPE_E3"
data_path="clearscope_e3_merge_edge_data.pt"
ground_truth_path="../Ground_Truth/ground_truth_nids_clearscope.pt"
gpu_ids=(0 1 0 1)
parallel=4
counter=0

for seed in "${seeds[@]}"
do
  for epoch in "${epochs[@]}"
  do
    for layer in "${layers[@]}"
    do
      for lr_f in "${lr_fs[@]}"
      do
        for lr in "${learning_rates[@]}"
        do
          for mask_rate in "${mask_rates[@]}"
          do
            for replace_rate in "${replace_rates[@]}"
            do
              for num_hidden in "${num_hiddens[@]}"
              do
                for num_head in "${num_heads[@]}"
                do
                  for max_epoch_f in "${max_epoch_fs[@]}"
                  do
                    for weight_decay in "${weight_decays[@]}"
                    do
                      for weight_decay_f in "${weight_decay_fs[@]}"
                      do
                        for drop_edge_rate in "${drop_edge_rates[@]}"
                        do
                          for in_drop in "${in_drops[@]}"
                          do
                            for attn_drop in "${attn_drops[@]}"
                            do
                              for alpha_l in "${alpha_ls[@]}"
                              do
                                result_name=$(python3 -c "
lr_val = float('${lr}')
lsf_val = float('${lr_f}')
wd_val = float('${weight_decay}')
wdf_val = float('${weight_decay_f}')
print(f'CLEARSCOPE_E3_loss_sce_dim_${num_hidden}_nhd_${num_head}_nh_${mask_rate}_nl_${layer}_lr_{lr_val}_lsf_{lsf_val}_mp_${epoch}_mpf_${max_epoch_f}_wd_{wd_val}_wdf_{wdf_val}_gatedge_gat')
")
                                if [ -f "save_middle_results/${result_name}.pt" ]; then
                                  echo "Skipping ${result_name}"
                                  ((counter++))
                                  continue
                                fi

                                device=${gpu_ids[$counter % parallel]}
                                CUDA_VISIBLE_DEVICES=$device python main_transductive.py \
                                  --device 0 \
                                  --dataset $dataset \
                                  --mask_rate $mask_rate \
                                  --encoder "gatedge" \
                                  --decoder "gat" \
                                  --in_drop $in_drop \
                                  --attn_drop $attn_drop \
                                  --num_layers $layer \
                                  --num_hidden $num_hidden \
                                  --num_heads $num_head \
                                  --max_epoch $epoch \
                                  --max_epoch_f $max_epoch_f \
                                  --lr $lr \
                                  --weight_decay $weight_decay \
                                  --lr_f $lr_f \
                                  --weight_decay_f $weight_decay_f \
                                  --activation prelu \
                                  --optimizer adam \
                                  --drop_edge_rate "$drop_edge_rate" \
                                  --loss_fn "sce" \
                                  --seeds $seed \
                                  --replace_rate $replace_rate \
                                  --alpha_l $alpha_l \
                                  --linear_prob \
                                  --scheduler \
                                  --use_cfg \
                                  --data_path $data_path \
                                  --ground_truth_path $ground_truth_path \
                                  --raw_data_dir ../raw_data \
                                  >> clearscope_e3_sweep.log 2>&1 &

                                ((counter++))
                                if [ $((counter % parallel)) -eq 0 ]; then
                                  wait
                                fi

                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
wait
echo "SWEEP_COMPLETE: ${counter} configs"
