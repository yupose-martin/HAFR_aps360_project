# APS360 Note

## Files
* Dataset.py loading the dataset
* HAFR.py author's tensorflow implementation
* pytorch_HAFR.py pytorch implementation

## How to run
* download the Dataset to ..\Data\
```
python .\pytorch_HAFR.py --path ..\Data\ --pretrain 0 --reg 0.1 --reg_image 0.01 --reg_h 1 --reg_w 1
'''
## Environment (just in case)
* python 3.10.14
* torch 2.4.1

## training parameters
* just like in lab and tut, epoch, lr, batch are adjustable
* subset_size could be set small if you just want to test the process. But for training, don't set it. It will default to using all training datas.
(python .\pytorch_HAFR.py --path ..\Data\ --pretrain 0 --reg 0.1 --reg_image 0.01 --reg_h 1 --reg_w 1 --subset_size 20)
* more parameters could be seen in pytorch_HAFR.py:args

## to do
* compute and store validation loss w.r.t epoch




# Author's README
# HAFR
This our official implementation for the paper: Hierarchical Attention Network for Visually-aware
Food Recommendation

If you use the codes, please kindly cite our paper. Thanks!

## Environment
Python 2.7 <br>
TensorFlow >= 1.4.0

## Quick Start
This command shows the effect of HAFR on pretrained model for dataset in epoch 300. <br>
```
python HAFR.py --pretrain 1 --reg 0.1 --reg_image 0.01 --reg_h 1 --reg_w 1 
```

## Dataset
We provide processed dataset: Allrecipes in [here](https://pan.baidu.com/s/1-CNkfmHL9kojlE1jIa3bJQ&shfl=sharepset) (password: sx7w) <br>

**data.train.rating** 
* Train file.
* Each line is a training instance: userID\t itemID\t rating

**data.test.rating**
* Test file.
* Each line is a testing instance: userID\t itemID\t rating

**data.test.negative**
* Test file (negative instances).
* Each line corresponds to the user of test.rating, containing 500 negative samples.
* Each line is in the format: (userID:itemID1,itemID2...)\t negativeItemID1\t negativeItemID2...

**data.valid.rating**
* Valid file.
* Each line is a validation instance: userID\t itemID\t rating

**data.valid.negative**
* Valid file (negative instances).
* Each line corresponds to the user of valid.rating, containing 500 negative samples.
* Each line is in the format: (userID:itemID1,itemID2...)\t negativeItemID1\t negativeItemID2...


