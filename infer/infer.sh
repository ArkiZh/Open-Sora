#!/bin/bash

# 指定你要遍历的文件夹路径
dir_path="/slurmhome/kzhang/repos/Open-Sora/run/outputs/STDiT2-XL-2-002"

# # 遍历文件夹下的所有目录
# for directory in $dir_path/*/
# do
#     # 对每个目录执行 python app.py 命令
#     echo "python inference-long.py sample_trained.py --ckpt-path ${directory}model"
#     CUDA_VISIBLE_DEVICES=2 python inference-long.py sample_trained.py --ckpt-path ${directory}model
# done

# 设置你想要的 CUDA_VISIBLE_DEVICES 值
cuda_visible_devices="2"

# # 遍历文件夹下的所有第一层目录
# for directory in $dir_path/*/
# do
#     # 只处理名称包含 "epoch68" 或 "epoch69" 的目录
#     if [[ $directory == *"epoch78"* || $directory == *"epoch69"* ]]; then
#         # 如果 model 目录存在
#         if [ -d "${directory}model" ]; then
#             # 设置 CUDA_VISIBLE_DEVICES 环境变量，并执行 python app.py 命令
#             echo "CUDA_VISIBLE_DEVICES=$cuda_visible_devices python inference-long.py sample_trained.py --ckpt-path ${directory}model"
#             # CUDA_VISIBLE_DEVICES=$cuda_visible_devices python inference-long.py sample_trained.py --ckpt-path ${directory}model
#         fi
#     fi
# done


declare -a epochs
# for i in {135..166}
# for i in {180..181}
for i in {269..270}
do
  epochs+=("epoch$i")
#   echo ${epochs[@]}
done

# declare -a epochs=("epoch71" "epoch72" "epoch73" "epoch74" "epoch 75" "epoch76" "epoch77" "epoch78")

# 遍历文件夹下的所有第一层目录
for directory in $dir_path/*/
do
    # 只处理名称包含 epochs 数组中任一元素的目录
    for epoch in "${epochs[@]}"
    do
        if [[ $(basename $directory) == *$epoch* ]]; then
            # 如果 model 目录存在
            if [ -d "${directory}model" ]; then
                # 设置 CUDA_VISIBLE_DEVICES 环境变量，并执行 python app.py 命令
                echo "CUDA_VISIBLE_DEVICES=$cuda_visible_devices python inference-long.py sample_trained.py --ckpt-path ${directory}model"
                CUDA_VISIBLE_DEVICES=$cuda_visible_devices python inference-long.py sample_trained.py --ckpt-path ${directory}model
            fi
        fi
    done
done