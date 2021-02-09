# DCRNet: Accelerating Quantitative Susceptibility Mapping using Compressed Sensing and Deep Neural Network

* This reposiotry is for a deep complex residual network (DCRNet) to recover both MR magnitude and quantitative phase images from the CS undersample k-space data, enabling the acceleration of QSM acquisitions, which is introduced in the following paper: xxxx.  

- This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and windows 10/ubuntu 19.10 with GTX 1060. 

# Content
- [ Overview](#head1)
	- [(1) Overall Framework](#head2)
	- [(2) Data Flow in Networks](#head3)
- [ Manual](#head4)
	- [Requirements](#head5)
	- [Quick Start (using example data)](#head6)
	- [The Whole Reconstruction Pipeline (on your own data)](#head7)
	- [Train new DCRNet](#head8)

# <span id="head1"> Overview </span>

## <span id="head2">(1) Overall Framework </span>

![Whole Framework](https://github.com/YangGaoUQ/DCRNet/blob/main/img/Figs_1.png)
Fig. 1: Overview of the proposed QSM accelerating scheme.  

## <span id="head3">(2) Data Flow in Networks </span>

![Data Flow](https://github.com/YangGaoUQ/DCRNet/blob/main/img/Figs_2.png)
Fig. 2: The architecture of the proposed DCRNet, which is developed from a deep residual network backbone using complex convolutional operations.

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

* For DL-based Magnitude and Phase Reconstruction  
    - Python 3.7 or later  
    - NVDIA GPU (CUDA 10.0)  
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.8 
    - MATLAB 2017b or later  
* For QSM PostProcessing from MRI Phase Data  
    - Hongfu Sun's QSM toolbox (https://github.com/sunhongfu/QSM)
    - FMRIB Software Library v6.0 (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)


## <span id="head6"> Quick Start (using example data) </span>
1. Clone this repository and make sure you have installed all prerequisites. 

```
    git clone https://github.com/YangGaoUQ/DCRNet.git
```
2. Download Exampledata provided by the authors from google drive, then unzip to get the Example Data.  

3. Run the following matlab script (in Folder './MatlabCodes/') to have a quick test on the Example Data.  

```matlab
    matlab -r "Demo_on_ExampleData.m"
```

## <span id="head7"> The Whole Reconstruction Pipeline (on your own data) </span>
1. Preprocess your test data, using 'Prepare_TestData.m' provided in the folder './TestData/'. 
```matlab 
    matlab -r "Prepare_TestData.m"
```

2. Modify the  test code. 
    1. Open ./Inference/Evaluate_set5.py using your own IDE
    2. go to line 37, set File_No = numer_of_your_own_images
    3. go to line 38, change 'set5' to your own directory
    4. save it as your own inference script file. 

3. Run the modified code
```python
    python your_own_inference_script.py  
```

## <span id="head8"> Train new DCRNet </span>
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
