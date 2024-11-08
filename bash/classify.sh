#!/bin/bash
python -m scripts.label_images_clevr_pos num_constraints=1 +image_dir=ComposableDiff/output/test_clevr_pos_5000_1
python -m scripts.label_images_clevr_pos num_constraints=2 +image_dir=ComposableDiff/output/test_clevr_pos_5000_2
python -m scripts.label_images_clevr_pos num_constraints=3 +image_dir=ComposableDiff/output/test_clevr_pos_5000_3
python -m scripts.label_images_clevr_pos num_constraints=4 +image_dir=ComposableDiff/output/test_clevr_pos_5000_4
python -m scripts.label_images_clevr_pos num_constraints=5 +image_dir=ComposableDiff/output/test_clevr_pos_5000_5
