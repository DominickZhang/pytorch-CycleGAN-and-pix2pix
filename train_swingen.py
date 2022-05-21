from utils import parse_args, get_rank, setup_for_distributed, is_main_process
from utils import create_logger, AverageMeter, get_grad_norm, load_checkpoint
import os
import datetime
from models.swin_transformer import SwinGenerator
from torch.utils.data import Dataset
import h5py as h5
import torch
from torchvision import transforms
from PIL import Image
import timm
import time
from termcolor import colored
import torch.distributed as dist
import numpy as np
from timm.utils.model import unwrap_model

class BratsDatasetHDF5(Dataset):
	def __init__(self, datapath, transform=None, transform_target=None, cross_validation_index=0, key='train'):
		super(BratsDatasetHDF5, self).__init__()

		self.datapath = datapath
		self.transform = transform
		self.transform_target = transform_target
		self.cross_validation_index = cross_validation_index
		self.key = key

		self.file = h5.File(datapath, "r", libver="latest", swmr=True)

	def __len__(self):
		if self.key is None:
			return len(self.file['data'])
		else:
			return len(self.file[self.key][0])

	def __getitem__(self, idx):
		if self.key is None:
			index = idx
		else:
			index = self.file[self.key][self.cross_validation_index][idx]
		img = self.file['data'][index]
		target = self.file['label'][index]

		if self.transform is not None:
			img = self.transform(img)

		if self.transform_target is not None:
			target = self.transform(target)

		return img, target

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

def create_model(img_size=256):
	return SwinGenerator(
		img_size=img_size,
		window_size=int(img_size/32),
		in_chans = 1,
		out_ch=1,
		)


def build_loader(datapath, key='train', cross_validation_index=0, resize_im=None,
				batch_size=32, num_workers=8):
	transform = None
	if resize_im is not None:
		transform = []
		transform.append(transforms.ToTensor())
		transform.append(transforms.Resize((resize_im, resize_im), interpolation=transforms.InterpolationMode.BICUBIC))
		transform = transforms.Compose(transform)

	dataset = BratsDatasetHDF5(datapath, 
									transform=transform, transform_target=transform,
									cross_validation_index=cross_validation_index, key=key)
	num_tasks = dist.get_world_size()
	global_rank = dist.get_rank()
	if key == 'train':
		sampler = torch.utils.data.DistributedSampler(
			dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False,
		)
		drop_last = True
	else:
		indices = np.arange(dist.get_rank(), len(dataset), dist.get_world_size())
		sampler = SubsetRandomSampler(indices)
		drop_last = False
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,drop_last=drop_last
		)
	return data_loader, dataset

def build_optimizer(model, optimizer_name='adamw', base_lr=1e-4, weight_decay=0.0):
	skip = {}
	skip_keywords = {}
	if hasattr(model, 'no_weight_decay'):
		skip = model.no_weight_decay()
	if hasattr(model, 'no_weight_decay_keywords'):
		skip_keywords = model.no_weight_decay_keywords()
	parameters = set_weight_decay(model, skip, skip_keywords)
	opt_lower = optimizer_name.lower()
	if opt_lower == 'sgd':
		ptimizer = torch.optim.SGD(parameters, momentum=0.9, nesterov=True,
					lr=base_lr, weight_decay=weight_decay)
	elif opt_lower == 'adamw':
		optimizer = torch.optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
					lr=base_lr, weight_decay=weight_decay)
	elif opt_lower == 'adam':
		optimizer = torch.optim.Adam(parameters, lr=base_lr, weight_decay=weight_decay, amsgrad=True)
	return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def get_stats(output):
	checkpoints = os.listdir(output)
	checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
	print(f"All checkpoints found in {output}: {checkpoints}")

	min_loss = 1.0e10
	max_loss = -1.0e10
	min_file = None
	max_file = None
	num_ckpts = len(checkpoints)
	for ckpt in checkpoints:
		ckpt_path = os.path.join(output, ckpt)
		ckpt_value = torch.load(ckpt_path, map_location='cpu')
		loss = ckpt_value['loss']
		if min_loss > loss:
			min_loss = loss
			min_file = ckpt_path
		if max_loss < loss:
			max_loss = loss
			max_file = ckpt_path
	return max_loss, max_file, min_loss, min_file, num_ckpts


def save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger):
	save_state = { 'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'opt_metric': opt_metric,
				'loss': loss,
				'epoch': epoch,
				'args': args,
	}
	save_path = os.path.join(args.output, f'ckpt_epoch_{epoch}.pth')
	logger.info(f'{save_path} saving......')
	for _ in range(200):
		try:
			torch.save(save_state, save_path)
			logger.info(f"{save_path} saved !!!")
			break
		except:
			logger.info(f"Save file {save_path} failed! re-try after 3 seconds")
			time.sleep(3)

def main():
	args = parse_args()
	output = args.output
	datapath = args.data_path
	datapath_test = args.data_path_test
	cross_validation_index = args.cross_validation_index
	img_size = args.img_size
	batch_size = args.batch_size
	total_epoch = args.total_epoch
	save_max = args.save_max
	base_lr = args.base_lr
	weight_decay = args.weight_decay
	patience = args.patience
	

	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ['RANK'])
		world_size = int(os.environ['WORLD_SIZE'])
		print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
	else:
		rank = -1
		world_size = -1

	try:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
		torch.distributed.barrier()
	except:
		pass

	os.makedirs(output, exist_ok=True)	
	logger = create_logger(args.output, dist_rank=get_rank(), name="Swin Generator")
	setup_for_distributed(is_main_process(), logger=logger)

	data_loader_train, _ = build_loader(datapath, key='train',
										cross_validation_index = cross_validation_index,
										resize_im = img_size, batch_size=batch_size)
	data_loader_val, dataset_val = build_loader(datapath, key='validation',
										cross_validation_index = cross_validation_index,
										resize_im = img_size, batch_size=batch_size)
	if datapath_test:
		data_loader_test, dataset_test = build_loader(datapath_test, key=None,
										resize_im = img_size, batch_size=batch_size)
	else:
		data_loader_test = None
		dataset_test = None


	model = create_model(img_size=img_size)
	model.cuda()
	optimizer = build_optimizer(model, optimizer_name='adam', base_lr=base_lr, weight_decay=weight_decay)
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
	model_without_ddp = unwrap_model(model)

	#criterion = torch.nn.MSELoss()
	criterion = torch.nn.L1Loss()

	opt_metric = 1.0e5
	resume_file = auto_resume_helper(output)
	if resume_file:
		if args.resume:
			logger.warning(f"auto-resume changing resume file from {args.resume} to {resume_file}")
		logger.info(f"auto resuming from {resume_file}")
		args.resume = resume_file
	else:
		logger.info(f"no checkpoint found in {output}, ignoring auto resume")

	if args.resume:
		opt_metric = load_checkpoint(args, model, optimizer, logger)

	if args.resume:
		loss = validate(model, criterion, data_loader_val, logger)
		logger.info(f"{args.resume}")
		logger.info(f"Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}%")

		if data_loader_test is not None:
			loss = validate(model, criterion, data_loader_test, logger)
			logger.info(f"Loss of the network on the {len(dataset_test)} test images: {loss:.5f}%")

	if args.eval:
		return


	logger.info("Start training")
	start_time = time.time()
	start_epoch = args.start_epoch
	patience_count = 0
	loss_buffer = 1.0e10
	for epoch in range(start_epoch, total_epoch):
		data_loader_train.sampler.set_epoch(epoch)

		train_one_epoch(args, model, criterion, data_loader_train, optimizer, epoch, logger)
		loss = validate(model, criterion, data_loader_val, logger)
		max_loss, max_file_path, _, _, num_ckpts = get_stats(output=output)
		if num_ckpts < save_max:
			save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger)
		elif max_loss > loss:
			os.remove(max_file_path)
			save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger)
		else:
			logger.info(f"Skip saving ckpt_epoch_{epoch}.pth...")
		logger.info(f"Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}%")
		opt_metric = min(opt_metric, loss)
		logger.info(f"Optimal Metric: {opt_metric:.5f}%")
		if loss >= loss_buffer:
			patience_count += 1
			if patience_count > patience:
				logger.info(f"Validation loss has not decreased for {patience} epochs")
				logger.info("Early stopping......")
				break
		else:
			patience_count = 0
		loss_buffer = loss

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	logger.info(f"Training time {total_time_str}")

	if data_loader_test is not None:
		logger.info("Start testing")
		loss = validate(model, criterion, data_loader_test, logger)
		logger.info(f"Loss of the network on the {len(dataset_test)} test images: {loss:.5f}%")

def train_one_epoch(args, model, criterion, data_loader, optimizer, epoch, logger):
	model.train()
	optimizer.zero_grad()

	num_steps = len(data_loader)
	batch_time = AverageMeter()
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()

	start = time.time()
	end = time.time()

	for idx, (samples, targets) in enumerate(data_loader):
		samples = samples.cuda(non_blocking=True)
		targets = targets.cuda(non_blocking=True)

		outputs = model(samples)
		loss = criterion(outputs, targets)
		optimizer.zero_grad()
		loss.backward()
		grad_norm = get_grad_norm(model.parameters())
		optimizer.step()

		torch.cuda.synchronize()

		loss_meter.update(loss.item(), targets.size(0))
		if grad_norm is not None:
			norm_meter.update(grad_norm)
		batch_time.update(time.time() - end)

		end = time.time()

		if idx % 10 == 0:
			memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
			etas = batch_time.avg * (num_steps - idx)
			logger.info(
				f'Train: [{epoch}/{args.total_epoch}][{idx}/{num_steps}]\t'
				f'eta {datetime.timedelta(seconds=int(etas))} lr {args.base_lr:.6f} \t'
				f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
				f'loss {loss_meter.val:.4f} ({loss_meter.avg:.5f})\t'
				f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
				f'mem {memory_used:.0f}MB')

	epoch_time = time.time() - start
	loss_meter.sync()
	logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}\t"
				f"train-loss {loss_meter.avg:.5f}")

@torch.no_grad()
def validate(model, criterion, data_loader, logger):
	model.eval()

	batch_time = AverageMeter()
	loss_meter = AverageMeter()

	end = time.time()
	for idx, (samples, targets) in enumerate(data_loader):
		samples = samples.cuda(non_blocking=True)
		targets = targets.cuda(non_blocking=True)

		outputs = model(samples)

		loss = criterion(outputs, targets)

		loss_meter.update(loss.item(), targets.size(0))

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % 10 == 0:
			memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
			info = (
				f'Test: [{idx}/{len(data_loader)}]\t'
				f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				f'Loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
				f'Mem {memory_used:.0f}MB'
				)
			logger.info(info)

	loss_meter.sync()

	logger.info(f' * Loss {loss_meter.avg:.5f}')
	return loss_meter.avg


if __name__ == '__main__':
	main()

	# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1/brats/
	# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val1/brats/ --cross_validation_index 1
	# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val5/brats/ --cross_validation_index 5
	# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val6/brats/ --cross_validation_index 6
	# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val7/brats/ --cross_validation_index 7
