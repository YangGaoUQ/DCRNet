# DCRNet: Accelerating Quantitative Susceptibility Mapping using Compressed Sensing and Deep Neural Network

* This reposiotry is for a deep complex residual network (DCRNet) to recover both MR magnitude and quantitative phase images from the CS undersample k-space data, enabling the acceleration of QSM acquisitions, which is introduced in the following paper: xxxx.  

- This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and Windows 10 environment with GTX 1060. 


# Content
- [ Overview](#head1)
	- [(1) Overall Framework](#head2)
	- [(2) Data Flow in Networks](#head3)
- [ Manual](#head4)
	- [ Requirements](#head5)
	- [Quick Test (inference on Set 5)](#head6)
	- [The whole test pipeline (on your own data)](#head7)
	- [Train new AutoBCS Net](#head8)
# <span id="head1"> Overview</span>

## <span id="head2">(1) Overall Framework</span>

![Whole Framework](https://github.com/YangGaoUQ/DCRNet/blob/main/img/Figs_1.png)
Fig. 1: Overview of the proposed QSM accelerating scheme.  

## <span id="head3">(2) Data Flow in Networks</span>

![Data Flow](https://github.com/YangGaoUQ/DCRNet/blob/main/img/Figs_2.png)
Fig. 2: The architecture of the proposed DCRNet, which is developed from a deep residual network backbone using complex convolutional operations.

# <span id="head4"> Manual</span>

## <span id="head5"> Requirements</span>
Python 3.7 or later  
NVDIA GPU (CUDA 10.0)  
Pytorch 1.8  
MATLAB 2017b or later  
Hongfu Sun's QSM toolbox (https://github.com/sunhongfu/QSM)


## <span id="head6">Quick Test (inference on Set 5)</span>
1. Clone this repository

```
    git clone https://github.com/YangGaoUQ/DCRNet.git
```

2. Run the following scripts (in Folder './Inference/') to test the pre-trained models.

```python
    python Evaluate_set5.py
```

## <span id="head7">The whole test pipeline (on your own data)</span>
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
## <span id="head8">Train new AutoBCS Net</span>
1. prepare your own trianing datasets (We used BSD500 database https://github.com/BIDS/BSDS500 )

2. Preprocessing data sets using the codes in the directory './Preprocessing_for_training' with Matlab
```matlab 
    matlab -r "GenerateData_model_64_96_Adam.m"
```

3. Enter the tranining folder ('./Training/'), and run the code: 
```python 
    python TrainAutoBCS.py 
```

[⬆ top](#readme)
