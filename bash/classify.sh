#!/bin/bash
# python -m scripts.label_images_clevr_pos num_constraints=1 +image_dir=ComposableDiff/output/test_clevr_pos_5000_1
# python -m scripts.label_images_clevr_pos num_constraints=2 +image_dir=ComposableDiff/output/test_clevr_pos_5000_2
# python -m scripts.label_images_clevr_pos num_constraints=3 +image_dir=ComposableDiff/output/test_clevr_pos_5000_3
# python -m scripts.label_images_clevr_pos num_constraints=4 +image_dir=ComposableDiff/output/test_clevr_pos_5000_4
# python -m scripts.label_images_clevr_pos num_constraints=5 +image_dir=ComposableDiff/output/test_clevr_pos_5000_5

# python -m scripts.label_images_clevr_pos num_constraints=1 +image_dir=runs/11-08_04-19-26_rejection_clevr_pos_1/test_clevr_pos_5000_1
python -m scripts.label_images_clevr_pos num_constraints=2 +image_dir=runs/11-07_23-10-44_rejection_clevr_pos_2/test_clevr_pos_5000_2
python -m scripts.label_images_clevr_pos num_constraints=3 +image_dir=runs/11-07_16-30-22_rejection_clevr_pos_3/test_clevr_pos_5000_3
python -m scripts.label_images_clevr_pos num_constraints=4 +image_dir=runs/11-07_08-06-39_rejection_clevr_pos_4/test_clevr_pos_5000_4
# python -m scripts.label_images_clevr_pos num_constraints=5 +image_dir=runs/rejection_clevr_pos_5/test_clevr_pos_5000_5
