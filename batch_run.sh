#!/bin/bash

python -m scripts.run_clevr --config-name=clevr_rel num_constraints=2 method_name=rejection \
max_samples_for_generation=10 experiment_name=base rejection.eval_batch_size=100

python -m scripts.run_clevr --config-name=clevr_rel num_constraints=2 method_name=rejection \
max_samples_for_generation=10 experiment_name=no_cfg rejection.eval_batch_size=100 elbo.use_cfg=False

python -m scripts.run_clevr --config-name=clevr_rel num_constraints=2 method_name=rejection \
max_samples_for_generation=10 experiment_name=random_noise rejection.eval_batch_size=100 elbo.same_noise=False elbo.sample_timesteps=random
