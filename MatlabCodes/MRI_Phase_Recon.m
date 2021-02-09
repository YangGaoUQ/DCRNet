
clear
clc

%% After Python Reconstruction:
% PostProcessing: save MRI magnitude and phase images;
addpath ./utils
SourceDir = '../TestData/'; %% make it the same as the MaskDir;
FileNo = 1;  %% file indentifier, the same as the one used in 'kSpace_Subsampling.m'
PhaseDir = '../MRI_QSM_recon/ExampleData/';  % you can modify it to be your own directory;
vox = [1 1 1]; % voxel size;
PyFolder = '../PythonCodes/';

%% load reconstruction data;
recon_r_path = [SourceDir,'rec_Input_',num2str(FileNo), '_real.mat'];
recon_i_path = [SourceDir,'rec_Input_',num2str(FileNo), '_imag.mat'];

load(recon_r_path);
load(recon_i_path);

%% load amplitude normlaization factors;
load(['Amp_Nor_factors_', num2str(FileNo),'.mat']);

%% postprocessing starts;
recs = recons_r + 1j * recons_i;

recs_new = zeros(size(recs));

[ny, nz, nx, ne] = size(recs); % image size;

for m = 1 : ne % from echo 1 to echo ne;
    rec_tmp = recs(:,:,:,m);  % reconstruction by DCRNet;
    recs_new(:,:,:,m) = Amp_Nor_factors(m) * rec_tmp * 30; % inverse the amplitude normlization;
end

%% save magnitude and phase images;
nii = make_nii(abs(recs_new), vox);
save_nii(nii, [PhaseDir, 'rec_Input_',num2str(FileNo),'_mag.nii']);

nii = make_nii(angle(recs_new), vox);
save_nii(nii, [PhaseDir, 'rec_Input_',num2str(FileNo),'_ph.nii']);



