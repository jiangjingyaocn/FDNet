# This code is primarily based on the VST project (https://github.com/nnizhang/VST/tree/main?tab=readme-ov-file).
# Only minor modifications have been made by the FDNet team.
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import get_loader
import math
from Models.ImageDepthNet import ImageDepthNet
import os
from torch import Tensor
import numpy as np

import torch.nn.functional as F
# from counter import Counter

# counter = Counter()


class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

    def forward(self, x):
        sobel_kernel_x = self.sobel_kernel_x.to(x.device)
        sobel_kernel_y = self.sobel_kernel_y.to(x.device)
        grad_x = F.conv2d(x, sobel_kernel_x, padding=1)
        grad_y = F.conv2d(x, sobel_kernel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad_magnitude

def get_contour(clone_gt):
    sobel_layer = SobelLayer().cuda()
    target_edges = sobel_layer(clone_gt)

    # 归一化到0到1之间
    target_edges_normalized = (target_edges - target_edges.min()) / (target_edges.max() - target_edges.min())
    threshold = 0.1
    # 二值化
    target_edges_binary = (target_edges_normalized > threshold).float()
    # target_edges_binary = target_edges
    # 转换为 PIL 图像
    # to_pil = transforms.ToPILImage()
    # # 保存每一张图片到指定路径
    # for i in range(target_edges_binary.size(0)):
    #     target_edges_cpu = target_edges_binary[i]
    #     edge_image = to_pil(target_edges_cpu)
    #     output_path = f'./outputimage/contour/train/output_image_{i}.png'
    #     edge_image.save(output_path)
    #     print(f"Edge image saved to {output_path}")
    # for i in range(clone_gt.size(0)):
    #     clone_gt_cpu = clone_gt[i]
    #     clone_gt_cpu = to_pil(clone_gt_cpu)
    #     output_path = f'./outputimage/contour/contour_gt/output_image_{i}.png'
    #     clone_gt_cpu.save(output_path)
    #     print(f"Edge image saved to {output_path}")
    # print(target_edges_binary.shape)
    return target_edges_binary
class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.s_co_bce = 0
        self.s_co_ssim = 0
        self.s_ct_bce = 0
        self.s_bg_bce = 0
        self.s_iou = 0
        self.s_ct_iou = 0

    def reset_loss(self):
        self.s_co_bce = 0
        self.s_co_ssim = 0
        self.s_ct_bce = 0
        self.s_bg_bce = 0
        self.s_iou = 0
        self.s_ct_iou = 0

    def iou(self, pred, gt):
        pred = F.sigmoid(pred)
        N, C, H, W = pred.shape
        min_tensor = torch.where(pred < gt, pred, gt)
        max_tensor = torch.where(pred > gt, pred, gt)
        min_sum = min_tensor.view(N, C, H * W).sum(dim=2)
        max_sum = max_tensor.view(N, C, H * W).sum(dim=2)
        loss = 1 - (min_sum / max_sum).mean()
        return loss

    # def structure_loss(self, pred, gt):
    #     weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
    #     wbce = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')
    #     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    #
    #     pred = torch.sigmoid(pred)
    #     inter = ((pred * gt) * weit).sum(dim=(2, 3))
    #     union = ((pred + gt) * weit).sum(dim=(2, 3))
    #     wiou = 1 - (inter + 1) / (union - inter + 1)
    #     return (wbce + wiou).mean()

    def ssim(self, pred, gt) -> float:
        """
        Calculate the ssim score.
        """
        _EPS = np.spacing(1)
        _, _, h, w = pred.shape
        N = h * w

        x = torch.mean(pred)
        y = torch.mean(gt)

        sigma_x = torch.sum((pred - x) ** 2) / (N - 1)
        sigma_y = torch.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = torch.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def stage_loss(self, stage_co_pred, stage_ct_preds, stage_bg_pred,
                   co_gt, ct_gt, bg_gt, loss_w):
        pred_size = stage_co_pred.shape[2:]
        co_gt = F.interpolate(co_gt, size=pred_size, mode="nearest")
        # noco_gt = F.interpolate(noco_gt, size=pred_size, mode="nearest")
        ct_gt = F.interpolate(ct_gt, size=pred_size, mode="nearest")
        bg_gt = F.interpolate(bg_gt, size=pred_size, mode="nearest")

        self.s_co_bce += self.bce(stage_co_pred, co_gt) * loss_w
        self.s_co_ssim += self.ssim(stage_co_pred, co_gt) * loss_w
        self.s_ct_bce += self.bce(stage_ct_preds, ct_gt) * loss_w
        self.s_bg_bce += self.bce(stage_bg_pred, bg_gt) * loss_w
        self.s_iou += self.iou(stage_co_pred, co_gt) * loss_w
        self.s_ct_iou += self.iou(stage_ct_preds, ct_gt) * loss_w

    def __call__(self, result, co_gt: Tensor):
        self.reset_loss()

        co_gt[co_gt < 0.5] = 0.
        co_gt[co_gt >= 0.5] = 1.

        contour_gt = co_gt.clone()
        ct_gt = get_contour(contour_gt)

        bg_gt = 1 - co_gt

        predictions_saliency, predictions_contour, predictions_background = result

        stage_co_preds = predictions_saliency
        stage_ct_preds = predictions_contour
        stage_bg_preds = predictions_background

        loss_weights = [1, 0.8, 0.5, 0.5]
        stage_num = len(stage_co_preds)
        for i in range(stage_num):
            self.stage_loss(
                stage_co_preds[i], stage_ct_preds[i], stage_bg_preds[i],
                co_gt, ct_gt, bg_gt, loss_weights[i]
            )
        total_loss = self.s_co_bce + self.s_co_ssim + self.s_ct_bce + self.s_bg_bce + self.s_iou

        return total_loss



def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

def main(local_rank, num_gpus, args):

    # counter.print_with_counter("Training.py")

    local_rank = 0
    cudnn.benchmark = True
    # counter.print_with_counter(args.init_method)

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    net.train()
    net.cuda()

    # counter.print_with_counter("Training.py")

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # criterion = nn.BCEWithLogitsLoss()
    cri = Criterion()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)


    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))
        total_loss_list = []

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True)),  \
                                        Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                                                      Variable(contour_28.cuda()), \
                                                      Variable(contour_56.cuda()), Variable(contour_112.cuda())

            # outputs_saliency, outputs_contour = net(images)
            result = net(images)
            total_loss = cri(result, label_224)
            total_loss_list.append(total_loss.item())
            print('Whole iter step:{0} - epoch progress:{1}/{2} - total_loss:{3:.4f} '
                        '- s_co_bce:{4:.4f} - s_co_ssim:{5:.4f} - s_ct_bce:{6:.4f} - s_bg_bce: {7:.4f} - s_iou:{8:.4f} - s_ct_iou:{9:.4f}  '
                        ' batch_size: {10}'.format(whole_iter_num, epoch, args.epochs, total_loss.item(),
                                                   cri.s_co_bce, cri.s_co_ssim, cri.s_ct_bce, cri.s_bg_bce, cri.s_iou,
                                                   cri.s_ct_iou, images.shape[0]))

            # epoch_total_loss += total_loss.cpu().data.item()

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(), args.save_model_dir + 'FDNet.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        loss_dir = 'loss'
        os.makedirs(loss_dir, exist_ok=True)
        plt.plot(total_loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title('Training Loss Over Time')
        # 保存图像到文件
        plt.savefig(os.path.join(loss_dir, f'epoch_{epoch}_loss.png'))
        plt.close()
        #
        # save_lossdir = './loss.txt'
        # save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)






