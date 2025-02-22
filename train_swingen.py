## Do not support multi-gpu training for now!
# The following functions will occupy all GPUs
# DistributedDataParallel
# torch.distributed.barrier()
# torch.distributed.all_reduce()

from utils import parse_args, get_rank, setup_for_distributed, is_main_process
from utils import create_logger, AverageMeter, get_grad_norm, load_checkpoint
import os
import datetime
from models.swin_transformer import SwinGenerator, SwinGeneratorResidual
from monai.networks.nets import UNet
from torch.utils.data import Dataset
import h5py as h5
import torch
from torchvision import transforms
from PIL import Image
import timm
import time
from termcolor import colored
import torch.distributed as dist
import torch.backends.cudnn as cudnn
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
            return len(self.file[self.key][self.cross_validation_index])

    def __getitem__(self, idx):
        #idx = int(idx)
        if self.key is None:
            index = idx
        else:
            index = self.file[self.key][self.cross_validation_index][idx]
        index = int(index)
        img = self.file['data'][index].transpose((1,2,0))
        #target = self.file['label'][index]
        target = self.file['truth'][index].transpose((1,2,0))

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

class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

def create_model(model_name='swin_gen', img_size=256):
    if model_name == 'swin_gen':
        return SwinGenerator(
            img_size=img_size,
            window_size=int(img_size/32),
            in_chans = 1,
            out_ch=1,
            )
    elif model_name == 'unet_wide':
        base_channel = 96
        channels=(base_channel, base_channel*2, base_channel*4, base_channel*8, base_channel*16)
        strides = tuple([2]*(len(channels)-1))
        return UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=channels,
                strides=strides,
                num_res_units=2,
                )
    elif model_name == 'unet_deep':
        base_channel = 16
        channels=(base_channel, base_channel*2, base_channel*4, base_channel*8, base_channel*24, base_channel*36, base_channel*48, base_channel*64)
        strides = tuple([2]*(len(channels)-1))
        return UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=channels,
                strides=strides,
                num_res_units=2,
                )
    elif model_name == 'swin_gen_residual':
        return SwinGeneratorResidual(
            img_size=img_size,
            window_size=int(img_size/32),
            in_chans = 1,
            out_ch=1,
            )
    elif model_name == 'swin_gen_residual_dense':
        return SwinGeneratorResidual(
            img_size=img_size,
            window_size=int(img_size/32),
            in_chans = 1,
            out_ch=1,
            residual_dense=True,
            )
    elif model_name == 'swin_gen_residual_attn':
        return SwinGeneratorResidual(
            img_size=img_size,
            window_size=int(img_size/32),
            in_chans = 1,
            out_ch=1,
            residual_dense=True,
            is_attn_residual=True,
            )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


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
        sampler = SubsetSampler(indices)
        drop_last = False
        if key == 'test':
            batch_size = 182
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
    model_name = args.model_name
    

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1


    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = args.random_seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(output, exist_ok=True)  
    logger = create_logger(args.output, dist_rank=get_rank(), name="Swin Generator")
    setup_for_distributed(is_main_process(), logger=logger)

    data_loader_train, _ = build_loader(datapath, key='train',
                                        cross_validation_index = cross_validation_index,
                                        resize_im = img_size, batch_size=batch_size,
                                        num_workers=2)
    data_loader_val, dataset_val = build_loader(datapath, key='val',
                                        cross_validation_index = cross_validation_index,
                                        resize_im = img_size, batch_size=batch_size*2,
                                        num_workers=2)

    try:
        data_loader_test, _ = build_loader(datapath, key='test',
                                            cross_validation_index = cross_validation_index,
                                            resize_im = img_size, batch_size=batch_size,num_workers=2)
    except Exception as e:
        logging.info(f"Creating test data loader fails...{e}")
        data_loader_test = None

    model = create_model(model_name=model_name, img_size=img_size)
    logger.info(str(model))
    model.cuda()
    optimizer = build_optimizer(model, optimizer_name='adam', base_lr=base_lr, weight_decay=weight_decay)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    #model_without_ddp = unwrap_model(model)

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
        logger.info(f"Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}")

        if data_loader_test is not None:
            num_test_data, loss = test(args, model, criterion, data_loader_test, logger)
            logger.info(f"Loss of the network on the {num_test_data} test images: {loss:.5f}")

        # if datapath_test:
        #     num_test_data, loss = test(args, model, criterion, datapath_test, logger)
        #     logger.info(f"Loss of the network on the {num_test_data} test images: {loss:.5f}")

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

        logger.info(f"Saving ckpt_epoch_{epoch}.pth...")
        if num_ckpts < save_max + 1: # N_{save_max} + Current model is saved
            save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger)
        else:
            os.remove(max_file_path)
            save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger)
        # elif max_loss > loss:
        #     os.remove(max_file_path)
        #     save_checkpoint(args, epoch, model, opt_metric, loss, optimizer, logger)
        # else:
        #     logger.info(f"Skip saving ckpt_epoch_{epoch}.pth...")
        logger.info(f"Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}%")
        opt_metric = min(opt_metric, loss)
        logger.info(f"Optimal Metric: {opt_metric:.5f}")

        if data_loader_test is not None:
            num_test_data, loss_test = test(args, model, criterion, data_loader_test, logger)
            logger.info(f"Loss of the network on the {num_test_data} test images: {loss_test:.5f}")

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
        num_test_data, loss_test = test(args, model, criterion, data_loader_test, logger)
        logger.info(f"Loss of the network on the {num_test_data} test images: {loss_test:.5f}")

    # if datapath_test:
    #     logger.info("Start testing")
    #     num_test_data, loss = test(args, model, criterion, datapath_test, logger)
    #     logger.info(f"Loss of the network on the {num_test_data} test images: {loss:.5f}")


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

@torch.no_grad()
def test(args, model, criterion, data_loader, logger):
    model.eval()

    all_metrics_list = []

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    if args.save_preds:
        pred_list = []

    N = len(data_loader)
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        logger.info(f">>> Predicting the {idx}(out of {N}) the volume...")
        
        ## Otherwise OOM
        outputs = []
        batch_size = samples.shape[0]
        batch_seg = batch_size // 4
        outputs.append(model(samples[:batch_seg]))
        outputs.append(model(samples[batch_seg: batch_seg*2]))
        outputs.append(model(samples[batch_seg*2: batch_seg*3]))
        outputs.append(model(samples[batch_seg*3:]))
        outputs = torch.cat(outputs, axis=0)

        loss = criterion(outputs, targets)

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)

        if args.save_preds:
            pred_list.append(pred_2_CT(outputs).detach().cpu().numpy().transpose(1,2,3,0))

        metrics = get_all_metrics(pred_2_CT(targets), pred_2_CT(outputs))
        metrics = [metric.item() for metric in metrics]
        all_metrics_list.append(metrics)

        end = time.time()

        if idx % 2 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            info = (
                f'Test: [{idx}/{N}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
                f'Mem {memory_used:.0f}MB'
                )
            logger.info(info)

    loss_meter.sync()
    all_metrics = np.mean(all_metrics_list, axis=0)

    logger.info(f"\n* RMSE (Brain, Air, Bone, Soft): {all_metrics[0]:.2f}, {all_metrics[1]:.2f}, {all_metrics[2]:.2f}, {all_metrics[3]:.2f}\n"
        f"* MAE (Brain, Air, Bone, Soft): {all_metrics[4]:.2f}, {all_metrics[5]:.2f}, {all_metrics[6]:.2f}, {all_metrics[7]:.2f}\n"
        f"* Dice (Brain, Air, Bone, Soft): {all_metrics[8]:.4f}, {all_metrics[9]:.4f}, {all_metrics[10]:.4f}, {all_metrics[11]:.4f}\n"
        )

    if args.save_preds:
        logger.info(f"Saving predictions...")
        np.save(os.path.join(args.output, 'predictions.npy'), np.array(pred_list))

    return N, loss_meter.avg

# @torch.no_grad()
# def test(args, model, criterion, test_data_folder, logger):
#     model.eval()

#     test_data_path = os.path.join(test_data_folder, 'test_data_syn.npy')
#     test_label_path = os.path.join(test_data_folder, 'test_label_syn.npy')
#     assert os.path.isfile(test_data_path), f"The test data does not exist! {test_data_path}"
#     assert os.path.isfile(test_label_path), f"The test label does not exist! {test_label_path}"

#     test_data = np.load(test_data_path)  # N x Modality x H x W x C
#     test_label = np.load(test_label_path)  # N x 1 x H x W x C

#     transform = []
#     transform.append(transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC))
#     transform = transforms.Compose(transform)

#     all_metrics_list = []

#     batch_time = AverageMeter()
#     loss_meter = AverageMeter()
#     end = time.time()

#     if args.save_preds:
#         pred_list = []

#     N = test_data.shape[0]
#     for idx in range(N):
#         logger.info(f">>> Predicting the {idx}(out of {N}) the volume...")
#         samples = test_data[idx].transpose(3,0,1,2)
#         samples = transform(torch.tensor(samples)).cuda(non_blocking=True)
#         targets = torch.tensor(test_label[idx].transpose(3,0,1,2)).cuda(non_blocking=True)
#         outputs = model(samples)

#         loss = criterion(outputs, CT_2_label(targets))

#         loss_meter.update(loss.item(), targets.size(0))
#         batch_time.update(time.time() - end)

#         if args.save_preds:
#             pred_list.append(pred_2_CT(outputs).detach().cpu().numpy().transpose(1,2,3,0))

#         metrics = get_all_metrics(targets-1000.0, pred_2_CT(outputs))
#         metrics = [metric.item() for metric in metrics]
#         all_metrics_list.append(metrics)

#         end = time.time()

#         if idx % 2 == 0:
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             info = (
#                 f'Test: [{idx}/{N}]\t'
#                 f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                 f'Loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
#                 f'Mem {memory_used:.0f}MB'
#                 )
#             logger.info(info)

#     loss_meter.sync()
#     all_metrics = np.mean(all_metrics_list, axis=0)

#     logger.info(f"\n* RMSE (Brain, Air, Bone, Soft): {all_metrics[0]:.2f}, {all_metrics[1]:.2f}, {all_metrics[2]:.2f}, {all_metrics[3]:.2f}\n"
#         f"* MAE (Brain, Air, Bone, Soft): {all_metrics[4]:.2f}, {all_metrics[5]:.2f}, {all_metrics[6]:.2f}, {all_metrics[7]:.2f}\n"
#         f"* Dice (Brain, Air, Bone, Soft): {all_metrics[8]:.4f}, {all_metrics[9]:.4f}, {all_metrics[10]:.4f}, {all_metrics[11]:.4f}\n"
#         )

#     if args.save_preds:
#         logger.info(f"Saving predictions...")
#         np.save(os.path.join(args.output, 'predictions.npy'), np.array(pred_list))

#     return N, loss_meter.avg


def cal_mask_mse(y_true, y_pred, y_mask):
    se = (y_true - y_pred)**2
    return (se*y_mask).sum()*1.0/y_mask.sum()

def cal_mask_mae(y_true, y_pred, y_mask):
    ae = (y_true - y_pred).abs()
    return (ae*y_mask).sum()*1.0/y_mask.sum()

def cal_F1_score_volume(y_true, y_pred):
    return 2.0*(y_true*y_pred).sum()/(y_true.sum() + y_pred.sum())

def pred_2_CT(array):
    return (array + 0.5)*2500.0

def CT_2_label(array):
    return array/2500.0 - 0.5

def get_all_metrics(y_true, y_pred):
    mask_brain = y_true >= -500
    mask_air = y_true < -500
    mask_soft = (y_true > -500) * (y_true < 300)
    mask_bone = y_true > 500

    pred_brain = y_pred >= -500
    pred_air = y_pred < -500
    pred_soft = (y_pred > -500) * (y_pred < 300)
    pred_bone = y_pred > 500

    rmse_brain = cal_mask_mse(y_true, y_pred, mask_brain).sqrt()
    rmse_air = cal_mask_mse(y_true, y_pred, mask_air).sqrt()
    rmse_soft = cal_mask_mse(y_true, y_pred, mask_soft).sqrt()
    rmse_bone = cal_mask_mse(y_true, y_pred, mask_bone).sqrt()

    mae_brain = cal_mask_mae(y_true, y_pred, mask_brain)
    mae_air = cal_mask_mae(y_true, y_pred, mask_air)
    mae_soft = cal_mask_mae(y_true, y_pred, mask_soft)
    mae_bone = cal_mask_mae(y_true, y_pred, mask_bone)

    dice_brain = cal_F1_score_volume(mask_brain, pred_brain)
    dice_air = cal_F1_score_volume(mask_air, pred_air)
    dice_soft = cal_F1_score_volume(mask_soft, pred_soft)
    dice_bone = cal_F1_score_volume(mask_bone, pred_bone)
    return [rmse_brain, rmse_air, rmse_bone, rmse_soft, mae_brain, mae_air, mae_bone, mae_soft, dice_brain, dice_air, dice_bone, dice_soft]

if __name__ == '__main__':
    main()


