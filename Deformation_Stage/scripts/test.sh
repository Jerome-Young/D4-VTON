python -u test.py -b 16 --gpu 0 --name d4vton_deform --mode test \
--exp_name <unpaired-cloth-warp|cloth-warp> \
--dataroot <your_dataset_path> \
--image_pairs_txt <test_pairs_unpaired_1018.txt|test_pairs_paired_1018.txt> \
--ckpt_dir checkpoints/vitonhd_deformation.pt