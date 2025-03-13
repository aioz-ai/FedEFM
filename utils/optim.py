import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from timm import scheduler
from torch import optim

class PolynomialLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size,
        iter_warmup,
        iter_max,
        power,
        min_lr=0,
        last_epoch=-1,
    ):
        self.step_size = step_size
        self.iter_warmup = int(iter_warmup)
        self.iter_max = int(iter_max)
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        iter_cur = float(self.last_epoch)
        if iter_cur < self.iter_warmup:
            coef = iter_cur / self.iter_warmup
            coef *= (1 - self.iter_warmup / self.iter_max) ** self.power
        else:
            coef = (1 - iter_cur / self.iter_max) ** self.power
        return (lr - self.min_lr) * coef + self.min_lr

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def step_update(self, num_updates):
        self.step()

def create_optimizer(opt_args, model):
    return optim.create_optimizer(opt_args, model)
def create_scheduler(opt_args, optimizer):
    if opt_args.sched == "polynomial":
        lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler

def get_optimizer(optimizer_name, net, lr_initial=1e-3):
    """

    :param optimizer_name:
    :param net:
    :param lr_initial:
    :return:
    """
    if optimizer_name == "adam":
        return optim.AdamW([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    elif optimizer_name == "sgd":
        return optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr_initial)

    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, epoch_size):
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1/np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cyclic":
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=0.1)

    elif scheduler_name == "custom":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30*int(epoch_size), gamma=0.1)
    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

