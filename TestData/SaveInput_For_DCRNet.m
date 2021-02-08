function  SaveInput_For_DCRNet(dataPath,maskPath, FileNo)

% example usage: 
% SaveInput_For_DCRNet('./kspace.mat','./Real_Mask_Acc4_256_by_128.mat', 01)

% inputs variables descriptions:
% 1) dataPath: path for the fully sampled k-space data. Data loading with this path will
%    load a variable named 'k' into the workspace, which is the k-space
%    data with an image size: (Nx * Ny * Nz) * NE, where NE is the number
%    of echos, and Nx, Ny, Nz are the Matrix sizes (FOV size ./ resolution);
% 2) maskPath: path for k-space subsampling mask;
%    loading maskPath will introduce a variable 'mask' into the workspace,
%    which is the k-space subsampling patterns.
% 3) FileNo: File

% output descriptions:
%    Input_{FileNo}_img.mat (image-domain subsampled data) and
%    Input_{FileNo}_k.mat (undersampled kspace data) will be saved as network input;

load(dataPath); % load k-space data, which is stored in variable 'k';
k = permute(k, [1, 3, 2, 4]); % prepare to conduct subsampling in the ky-kz (coronal) plane

ll = size(k);

if length(ll) == 3
    ll = [ll, 1];
end

load(maskPath);

inputs_img = zeros(ll);% image-domain subsampled data (zero-filling reconstructions)
inputs_k = zeros(ll); % undersampled kspace data



for i = 1 : ll(4)
    k_full = k(:,:,:,i);
    
    k_full = k_full / max(abs(k_full(:))); %% amplitude normalization
    
    % subsample the fully-sampeld k-space data
    k_sub = k_full .* mask;
    
    % zero-filling reconstructions (image-domain inputs)
    img_tmp = fftn(fftshift(k_sub));
    
    % one dimensional fft on the kx (read out direction);
    % k-space data consistency inputs
    k_sub = fft(fftshift(k_sub, 3), [], 3);
    
    % slicing into 2D slices along kx direction;
    for j = 1 : ll(3)
        tmp = img_tmp(:,:, j);
        tmp = tmp / 30;  %% simple scaling normalization
        inputs_img(:, :, j, i) = tmp;
        
        tmp = k_sub(:,:, j);
        tmp = tmp / 30;  %% simple scaling normalization
        inputs_k(:, :, j, i) = tmp;
    end
end

save(['./Input_', num2str(FileNo),'_img.mat'], 'inputs_img');
save(['./Input_', num2str(FileNo),'_k.mat'], 'inputs_k');
end

