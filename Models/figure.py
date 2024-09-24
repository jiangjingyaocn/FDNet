import os
import torch
import argparse
from ImageDepthNet import ImageDepthNet

import socket
import torch.distributed as dist

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr, port = s.getsockname()
    s.close()
    return port

if __name__ == "__main__":
    counter = Counter()

    free_port = find_free_port()
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    counter.print_with_counter(f'port:  {free_port}')
    parser.add_argument('--init_method', default=f'tcp://127.0.0.1:{free_port}', type=str, help='init_method')
    parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
    parser.add_argument('--img_size', default=384, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/83.3_T2T_ViT_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')

    # test
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    # parser.add_argument('--test_paths', type=str, default='ECSSD')
    parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O+SOD')

    # evaluation
    parser.add_argument('--Evaluation', default=True, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='FDNet', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    model = ImageDepthNet
    dummy_input = torch.randn(1, 3,384,384 )

# 将模型转换为ONNX格式
    torch.onnx.export(model, (dummy_input,args), "VST.onnx", opset_version=11)
