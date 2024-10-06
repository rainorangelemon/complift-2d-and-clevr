#!/bin/bash

# ================================= CLEVR Relational Dataset=================================

# suppose the current directory is ComposableDiff/scripts
cd ..
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_rel_5000_1.npz'
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_rel_5000_2.npz'
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_rel_5000_3.npz'

# evaluate the model
cd classifier
python eval.py --dataset clevr_rel --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_rel_5000_1.npz  --generated_img_folder ../output/test_clevr_rel_5000_1 --mode generation
python eval.py --dataset clevr_rel --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_rel_5000_2.npz  --generated_img_folder ../output/test_clevr_rel_5000_2 --mode generation
python eval.py --dataset clevr_rel --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_rel_5000_3.npz  --generated_img_folder ../output/test_clevr_rel_5000_3 --mode generation


# ================================= CLEVR Position Dataset=================================
# suppose the current directory is ComposableDiff/classifier
cd ..
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma False --use_scale_shift_norm False --num_classes 2 --dataset clevr_pos --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_pos_5000_1.npz'
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_pos_5000_2.npz'
python -m scripts.image_sample_compose_clevr_rel $MODEL_FLAGS $DIFFUSION_FLAGS --data_path './dataset/test_clevr_pos_5000_3.npz'

# evaluate the model
cd classifier
python eval.py --dataset clevr_pos --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_pos_5000_1.npz  --generated_img_folder ../output/test_clevr_pos_5000_1 --mode generation
python eval.py --dataset clevr_pos --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_pos_5000_2.npz  --generated_img_folder ../output/test_clevr_pos_5000_2 --mode generation
python eval.py --dataset clevr_pos --checkpoint_dir ../models --im_size 128 --filter_dim 64  --npy_path ../dataset/test_clevr_pos_5000_3.npz  --generated_img_folder ../output/test_clevr_pos_5000_3 --mode generation

