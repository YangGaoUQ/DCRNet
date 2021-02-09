
clear
clc
close all

%% set parameters for saving the inputs data and subsampling mask;
MaskDir = '../TestData/';
FileNo = 1;
addpath ./utils;

%% data loading
k = load('../TestData/kspace_example.mat');
k = k{1};

%% generate subsampling mask (AF = 4)
k = permute(k, [1, 3, 2, 4]); %conduct subsampling in the ky-kz (coronal) plane

[ny, nz, nx, ne] = size(k); % image size;

[mask] = Gen_Sampling_Mask([ny, nz], 4, 12, 1.8); %

%% Subsampling the fully-sampled kspace data and save it in appropriate for DCRNet;
Amp_Nor_factors = Save_Input_Data_For_DCRNet(k, mask, FileNo, MaskDir);

save(['Amp_Nor_factors_', num2str(FileNo),'.mat'],'Amp_Nor_factors');

msg = msgbox('Go to Python Code Folder for DCRnet Reconstruction', 'Subsampling Completed', 'help');
waitfor(msg)
ConfigPython; 
!python ../PythonCodes/Eval_Batches.py





