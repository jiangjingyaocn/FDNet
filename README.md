# FDNet
Source code for FDNet as described in the paper.

## Environment 
The model is implemented using PyTorch in a CUDA11 environment and trained on an NVIDIA A100 80GB PCIe GPU
- python-3.8
- torch-2.0.0
- torchvision-0.15.1

## Training Set
We use the training set of [DUTS](http://saliencydetection.net/duts/).After Downloading, put them into FDNet/Data folder.
## Testing Set
We use the testing set of [DUTS](http://saliencydetection.net/duts/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](https://ccvl.jhu.edu/datasets/) and [DUT-O](http://saliencydetection.net/dut-omron/#orgc1e7c62) to test our model. After Downloading, put them into FDNet/Data folder.

Your FDNet/Data` folder should look like this:

```
-- Data
   |-- DUTS
   |   |-- DUTS-TR
   |   |-- | DUTS-TR-Image
   |   |-- | DUTS-TR-Mask
   |   |-- | DUTS-TR-Contour
   |   |-- DUTS-TE
   |   |-- | DUTS-TE-Image
   |   |-- | DUTS-TE-Mask
   |-- ECSSD
   |   |--images
   |   |--GT
   ...
```
## Training and Testing
- Download the pretrained T2T-ViT-14 model[[kuakepan](https://pan.quark.cn/s/87dc8d6a4d66)fetch code:DasR] and put it into `pretrained_model/` folder.
- Run `python train_test_eval.py --Training True --Testing True` for training, and testing. The predictions will be in `preds/` folder.
## Testing on Our Pretrained FDNet Model
Download our pretrained `FDNet.pth`[[kuakepan](https://pan.quark.cn/s/e406ae45daea)fetch code:jh6Q] and then put it in `checkpoint/` folder. 
Run `python train_test_eval.py --Testing True` for testing. The predictions will be in `preds/` folder
## Results
Our saliency maps can be downloaded from [kuakepan](https://pan.quark.cn/s/744882978ab7)fetch code:PQf9.
## Evaluation
We use [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit) as evaluation tool.
## Acknowledgments
- **VST Project**: The core of this project is largely derived from the [VST](https://github.com/nnizhang/VST) codebase. We have made minor adjustments for our specific use case.
- **Evaluation Tools**: We would also like to thank [lartpang](https://github.com/lartpang/PySODEvalToolkit) for providing the evaluation tools used in this project.
