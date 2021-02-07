# DCRNet: Accelerating Quantitative Susceptibility Mapping using Compressed Sensing and Deep Neural Network
This reposiotry is for a deep complex residual network (DCRNet) to recover both MR magnitude and quantitative phase images from the CS undersample k-space data, enabling the acceleration of QSM acquisitions, which is introduced in the following paper: xxxx.  

This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and Windows 10 environment with GTX 1060. 

# Overview



### (1) Overall Framework

![Whole Framework](https://github.com/YangGaoUQ/DCRNet/tree/main/img/Figs_1.png)
Fig. 1: Overview of the proposed QSM accelerating scheme.  

### (2) Data Flow in Networks

![Data Flow](https://github.com/YangGaoUQ/DCRNet/tree/main/img/Figs_2.png)
Fig. 2: The architecture of the proposed DCRNet, which is developed from a deep residual network backbone using complex convolutional operations.

## Requirements
Python 3.7 or later  
NVDIA GPU (CUDA 10.0)  
Pytorch 1.10 or later  
MATLAB 2017b or later  

# Manual

## Quick Test (inference on Set 5)
1. Clone this repository

```
    git clone https://github.com/YangGaoUQ/AutoBCS.git
```

2. Run the following scripts (in Folder './Inference/') to test the pre-trained models.

```python
    python Evaluate_set5.py
```

## The whole test pipeline (on your own data)
1. Prepare your test data, and make your own directory for it, and rename them in a numerical order. (You can use Prepare_TestData.m provided in the folder './set5/' to process your data.) 
```matlab 
    matlab -r "Prepare_TestData.m"
```

2. Modify the  test code. 
    1. Open ./Inference/Evaluate_set5.py using your own IDE
    2. go to line 37, set File_No = numer_of_your_own_images
    3. go to line 38, change 'set5' to your own directory
    4. save it as your own test script file. 

3. Run your own code

```python
    python your_own_test_script.py  
```
## Train new AutoBCS Net
1. prepare your own trianing datasets (We used BSD500 database https://github.com/BIDS/BSDS500 )

2. Preprocessing data sets using the codes in the directory './Preprocessing_for_training' with Matlab
```matlab 
    matlab -r "GenerateData_model_64_96_Adam.m"
```

3. Enter the tranining folder ('./Training/'), and run the code: 
```python 
    python TrainAutoBCS.py 
```



