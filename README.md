# [Adapting Unstructured Sparsity Techniques for Structured Sparsity]()
[Aditya Kusupati](https://homes.cs.washington.edu/~kusupati/)

This repository contains code for the CNN experiments presented in the [paper]() along with more functionalities.

This code base is built upon the [STR](https://github.com/RAIVNLab/STR) modified for [STR-BN]() experiments.

## Set Up
0. Clone this repository.
1. Using `Python 3.6`, create a `venv` with  `python -m venv myenv` and run `source myenv/bin/activate`. You can also use `conda` to create a virtual environment.
2. Install requirements with `pip install -r requirements.txt` for `venv` and appropriate `conda` commands for `conda` environment.
3. Create a **data directory** `<data-dir>`.
To run the ImageNet experiments there must be a folder `<data-dir>/imagenet`
that contains the ImageNet `train` and `val`.

## STR-BN
[`STR-BN`](utils/bn_type.py#L20). Users can take `STR-BN` and use it in most of the PyTorch based models as it inherits from `nn.BatchNorm2d` or also mentioned here as [`LearnedBatchNorm`](utils/bn_type.py#L6). The hyperparameters of `STR-BN` which includes the [`sparseFunction`](utils/bn_type.py#L8) are not well explored to provide the users with default settings. This is experimental code and contributions are welcome.

## Vanilla Training
This codebase contains model architectures for [ResNet18](models/resnet.py#L156), [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) and support to train them on ImageNet-1K. We have provided some `config` files for training [ResNet50](models/resnet.py#L161) and [MobileNetV1](models/mobilenetv1.py) which can be modified for other architectures and datasets. To support more datasets, please add new dataloaders to [`data`](data/) folder.

Training across multiple GPUs is supported, however, the user should check the minimum number of GPUs required to scale ImageNet-1K. 

### Train dense models on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-dense.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-dense.yaml --multigpu 0,1,2,3```

### Train models with **[STR-BN]()** on ImageNet-1K:

ResNet50: ```python main.py --config configs/largescale/resnet50-str-bn.yaml --multigpu 0,1,2,3```

MobileNetV1: ```python main.py --config configs/largescale/mobilenetv1-str-bn.yaml --multigpu 0,1,2,3```

The user can explore and search for right hyperparameters of `STR-BN` through the [`configs`](configs/).

## Sparsity Budgets

The folder [`budgets`](budgets) contains the csv files containing all the non-uniform sparsity budgets STR learnt for ResNet50 on ImageNet-1K across all the sparsity regimes along with baseline budgets for 90% sparse ResNet50 on ImageNet-1K. In case, you are not able to use the pretraining models to extract sparsity budgets, you can directly import the same budgets using these files. Structured sparsity methods which take in a layer-wise sparsity budget could potentially utilize these budgets learnt through STR for unstructured sparsity.

## Citation

If you find this project useful in your research, please consider citing:

```
@article{Kusupati20a
  author    = {Kusupati, Aditya},
  title     = {Adapting Unstructured Sparsity Techniques for Structured Sparsity},
  booktitle = {},
  year      = {2020},
}
```
