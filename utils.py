import argparse
import os
import sys
import logging
import functools
from termcolor import colored
import torch.distributed as dist
import timm
import torch

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def parse_args():
	parser = argparse.ArgumentParser(description="Swin Generator for Medical Imaging Synthesis")
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--data_path_test', type=str, default='')
	parser.add_argument('--cross_validation_index', type=int, default=0)
	parser.add_argument('--output', type=str, default='output/')
	parser.add_argument('--local_rank', type=int, default=0)
	args = parser.parse_args()
	return args

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def load_checkpoint(args, model, optimizer, logger):
	model_without_ddp = timm.unwrap_model(model)
	logger.info(f"=============> Resuming from {resume_file}...................")
	checkpoint = torch.load(args.resume, map_location='cpu')

	if 'model' in checkpoint:
		model.load_state_dict(checkpoint['model'], strict=True)

	if 'epoch' in checkpoint:
		args.start_epoch = checkpoint['epoch'] + 1

	if 'optimizer' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer'], strict=True)

	if 'opt_metric' in checkpoint:
		opt_metric = checkpoint['opt_metric']
	else:
		opt_metric = 0.0

	return opt_metric

def is_main_process():
    return dist.get_rank() == 0

def setup_for_distributed(is_master, logger = None):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            if logger is None:
                builtin_print(*args, **kwargs)
            else:
                if len(kwargs) == 0:
                    logger.info(', '.join(map(str, args)))
                else:
                    builtin_print(*args, **kwargs)

    __builtin__.print = print

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / max(1, self.count)

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt