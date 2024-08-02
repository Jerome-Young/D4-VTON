python test.py --gpu_id 0 --ddim_steps 100 \
--outdir results/d4vton_unpaired_syn --config configs/vitonhd_512.yaml \
--dataroot <your_dataset_path> \
--ckpt checkpoints/vitonhd_synthesis.ckpt --delta_step 89 \
--n_samples 12 --seed 23 --scale 1 --H 512 --unpaired