#!/bin/bash
python -m scripts.run_clevr num_constraints=1 experiment_name=rejection_clevr_pos_1
python -m scripts.run_clevr num_constraints=2 experiment_name=rejection_clevr_pos_2
python -m scripts.run_clevr num_constraints=3 experiment_name=rejection_clevr_pos_3
python -m scripts.run_clevr num_constraints=4 experiment_name=rejection_clevr_pos_4
