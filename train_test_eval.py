# This code is primarily based on the VST project (https://github.com/nnizhang/VST/tree/main?tab=readme-ov-file).
# Only minor modifications have been made by the FDNet team.
import os
import torch
import Training
import Testing
from Evaluation import main
import argparse

import socket
import torch.distributed as dist
from counter import Counter

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr, port = s.getsockname()
    s.close()
    return port


# 创建计数器对象
if __name__ == "__main__":
    counter = Counter()

    free_port = find_free_port()
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    counter.print_with_counter(f'port:  {free_port}')
    parser.add_argument('--init_method', default=f'tcp://127.0.0.1:{free_port}', type=str, help='init_method')
    parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=384, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/83.3_T2T_ViT_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=11, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='selectedall')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus =1
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
        torch.cuda.empty_cache()
    if args.Testing:
        Testing.test_net(args)
        torch.cuda.empty_cache()