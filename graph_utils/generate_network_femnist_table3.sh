#!/usr/bin/env bash
# med dataset - Amazon network
# >>> Baseline
# python generate_networks.py amazon_us --experiment med --upload_capacity 1e10 --download_capacity 1e10
# # >>> Multigraph RING
# python generate_network_multigraph.py amazon_us --experiment med --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# med dataset - Ebone network
# >>> Baseline
# python generate_networks.py ebone --experiment med --upload_capacity 1e10 --download_capacity 1e10
# # >>> Multigraph RING
# python generate_network_multigraph.py ebone --experiment med --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# med dataset - Gaia network
# >>> Baseline
python3 generate_networks.py gaia --experiment med --upload_capacity 1e10 --download_capacity 1e10
# >>> Multigraph RING
#python generate_network_multigraph.py gaia --experiment med --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# med dataset - Geant network
# >>> Baseline
# python3 generate_networks.py geantdistance --experiment med --upload_capacity 1e10 --download_capacity 1e10
# # >>> Multigraph RING
# python3 generate_network_multigraph.py geantdistance --experiment med --upload_capacity 1e10 --download_capacity 1e10 --arch ring

# # med dataset - Exodus network
# # >>> Baseline
# python generate_networks.py exodus --experiment med --upload_capacity 1e10 --download_capacity 1e10
# # >>> Multigraph RING
# python generate_network_multigraph.py exodus --experiment med --upload_capacity 1e10 --download_capacity 1e10 --arch ring