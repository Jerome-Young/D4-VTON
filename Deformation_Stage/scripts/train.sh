python -m torch.distributed.launch --nproc_per_node=4 --master_port=6231 train.py \
--dataroot <your_dataset_path> \
-b 2 --num_gpus 4 --name d4vton_deform --group_num 8