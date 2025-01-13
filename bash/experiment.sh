#!/bin/bash

# Commented out rejection experiments
# python -m scripts.run_clevr_rejection num_constraints=1 experiment_name=rejection_clevr_pos_1
# python -m scripts.run_clevr_rejection num_constraints=2 experiment_name=rejection_clevr_pos_2
# python -m scripts.run_clevr_rejection num_constraints=3 experiment_name=rejection_clevr_pos_3
# python -m scripts.run_clevr_rejection num_constraints=4 experiment_name=rejection_clevr_pos_4

# Run experiments with different samplers (ULA and UHMC)
for sampler in ULA UHMC; do
    for num_constraints in {1..5}; do
        python -m scripts.run_clevr_ebm \
            num_constraints=$num_constraints \
            experiment_name="ebm_${sampler,,}_clevr_pos_$num_constraints" \
            ebm.num_samples_per_trial=10 \
            ebm.sampler_type=$sampler
    done
done

