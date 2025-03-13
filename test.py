import os
from torch.multiprocessing import Process
import torch.distributed as dist
import torch

from utils.args import parse_args
from utils.utils import loggs_to_json, args_to_string
from communication import Peer2PeerNetwork
import time

def init_process(rank, size, arguments, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, arguments)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    total_train_time = 0.0
    print("Run experiment in sequential setting..")

    if args.architecture == "ring":
        network = Peer2PeerNetwork(args)
    else:
        print("Not Support yet!")
    network.load_models(args.n_rounds)
    network.write_logs()
