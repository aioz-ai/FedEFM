#!/usr/bin/env bash
python3 -c "
import torch 
torch.cuda.empty_cache()"

python3 main.py med --network_name gaia --architecture ring --n_rounds 6400 --bz_train 1 --bz_test 1 --device cuda:0 --log_freq 1 --local_steps 1 --lr 0.001 --decay sqrt 