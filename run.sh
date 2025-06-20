# source /home/lsl/anaconda3/envs/zeroRE/bin/activate zeroRE

# fewrel
# for dataset in 'wikizsl'
# do
#   for unseen in 15 
#   do
#     for seed in 7
#     do
#         for k in 2 
#         do
#         python -u main.py \
#         --gpu_available 0 \
#         --unseen ${unseen} \
#         --k ${k} \
#         --dataset ${dataset} \
#         --seed ${seed} \
#         --train_batch_size 32 \
#         --evaluate_batch_size 640 \
#         --epochs 5 \
#         --lr 2e-5 \
#         --warm_up 100 \
#         --pretrained_model_name_or_path ../../bert-base-uncased \
#         --add_auto_match True
#         done
#     done
#   done
# done

# 7 19 42 66 101 wikizsl
for dataset in 'fewrel'
do
  for unseen in 5
  do
    for seed in 7 
    do
        for k in 2  
        do
        python -u main.py \
        --gpu_available 0 \
        --unseen ${unseen} \
        --k ${k} \
        --dataset ${dataset} \
        --seed ${seed} \
        --train_batch_size 64 \
        --evaluate_batch_size 640 \
        --epochs 4 \
        --lr 2e-5 \
        --warm_up 100 \
        --des_pretrained_model_name_or_path ../bert-base-uncased \
        --input_pretrained_model_name_or_path ../Roberta 
        --add_auto_match True
        done
    done
  done
done