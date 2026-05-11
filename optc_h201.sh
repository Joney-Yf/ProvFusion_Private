#!/bin/bash

# 定义参数数组
learning_rates=(0.001 0.01 0.00001)  # lr 参数的不同取值
lr_fs=(0.0001 0.001 0.00005 0.00015)   # lr 参数的不同取值
mask_rates=(0.1 0.3 0.5)        # mask_rate 参数的不同取值
layers=(2)                          # layer 参数的不同取值
seeds=(1)                                 # seeds 参数的不同取值
epochs=(200 500)                                # 训练 epochs
num_hiddens=(64)                # num_hidden 参数的不同取值
replace_rates=(0.0)                    # num_decoder_layers 参数的不同取值
weight_decays=(5e-5 1e-5 1e-4 2e-6)  # weight_decay 参数的不同取值
weight_decay_fs=(5e-5)  # weight_decay 参数的不同取值
num_heads=(2 4 8)                         # num_heads 参数的不同取值
max_epoch_fs=(10)              # max_epoch_f 参数的不同取值
drop_edge_rates=(0.0)    # drop_edge_rate 参数的不同取值
in_drops=(0.2)                  # in_drop 参数的不同取值
attn_drops=(0.1)                  # attn_drop 参数的不同取值
alpha_ls=(3)                  # alpha_l 参数的不同取值
dataset="OPTC_h201"
dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
data_path="${dataset_lower}_merge_edge_normalized.pt"
ground_truth_path="../Ground_Truth/ground_truth_nids_${dataset_lower}.pt"
gpu_ids=(0)
counter=0
# 循环遍历所有参数组合
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
                for weight_decay in "${weight_decays[@]}"
                do
                  for weight_decay_f in "${weight_decay_fs[@]}"
                  do
                    for num_head in "${num_heads[@]}"
                    do
                      for max_epoch_f in "${max_epoch_fs[@]}"
                      do
                        for drop_edge_rate in "${drop_edge_rates[@]}"
                        do
                          for in_drop in "${in_drops[@]}"
                          do
                            for attn_drop in "${attn_drops[@]}"
                            do
                              for alpha_l in "${alpha_ls[@]}"
                              do
                                device=${gpu_ids[$counter % 1]}

                                # 运行 Python 脚本，替换对应的参数
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
                                  >> optc_log/optc_h201_$device.log  &  \

                                ((counter++))
                                if [ $((counter % 1)) -eq 0 ]; then
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