#!/usr/bin/env bash
#!/usr/bin/env bash
python test.py femnist --network_name exodus --architecture ring --n_rounds 6400 --bz_train 128 --bz_test 2048 --device cuda --log_freq 800 --local_steps 1 --lr 0.001 --decay sqrt --multigraph --test --save_logg_path pretrained_models/FEMNIST_EXODUS_MULTIGRAPH-RING