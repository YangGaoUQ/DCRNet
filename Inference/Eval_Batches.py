import torch 
import torch.nn as nn
import numpy as np
import scipy.io as scio
from Attentions import * 
import time 


if __name__ == '__main__':

    img_ids = [i_id.strip() for i_id in open("../input_files.txt")]
    k_ids = [i_id.strip() for i_id in open("../input_files_k.txt")]
    # print(self.img_ids)
    ## get all fil names, preparation for get_item. 
    ## for example, we have two files: 
    ## 102-field.nii for input, and 102-phantom for label; 
    ## then image id is 102, and then we can use string operation
    ## to get the full name of the input and label files. 
    files = []
    for name in img_ids:
        input_files = "/scratch/itee/yang_phase_acc/Make_Figures/" + name
        files.append({
            #"img": img_file,
            "inputs_img": input_files,
            "name": name
        })

    files_k = []
    for name in k_ids:
        input_files = "/scratch/itee/yang_phase_acc/Make_Figures/" + name
        files_k.append({
            #"img": img_file,
            "inputs_k": input_files,
            "name": name
        })


    with torch.no_grad():        
        print('Network Loading')
        ## load trained network 
        octnet = ReconNet(5)
        octnet = nn.DataParallel(octnet)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        octnet.load_state_dict(torch.load('./L2Loss_LS.pth'))
        octnet.to(device)
        octnet.eval()

        for idx in range(5, 6):

            matImage = scio.loadmat('Real_Mask_Acc4_256_by_128.mat')
            mask = matImage['mask']  
            mask = np.array(mask)
            mask = torch.from_numpy(mask)
            mask = mask.float()
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            print(mask.size())
            mask = mask.to(device)

            datafiles = files[idx]
            print(datafiles["inputs_img"])
            print('Loading Subsampled Data')   
            matTest = scio.loadmat(datafiles["inputs_img"], verify_compressed_data_integrity=False)
            image = matTest['inputs']
            image = np.array(image)

            image_r = np.real(image)
            image_i = np.imag(image)

            image_r = torch.from_numpy(image_r)
            image_i = torch.from_numpy(image_i)

            image_r = image_r.float()
            image_i = image_i.float()

            print(idx)
            datafiles_k = files_k[idx]
            print(datafiles_k["inputs_k"])
            print('Loading kspace Data')   
            matTest = scio.loadmat(datafiles_k["inputs_k"], verify_compressed_data_integrity=False)
            k0 = matTest['inputs_k']
            k0= np.array(k0)

            k0_r = np.real(k0)
            k0_i = np.imag(k0)

            k0_r = torch.from_numpy(k0_r)
            k0_i = torch.from_numpy(k0_i)

            k0_r = k0_r.float()
            k0_i = k0_i.float()

            ####image_r = torch.unsqueeze(image_r, 3) ##  1 * 256 * 256
            ####image_i = torch.unsqueeze(image_i, 3) ##  1 * 256 * 256

            recons_r = torch.zeros(image_r.size())
            recons_i = torch.zeros(image_i.size())

            print('reconing ...')
            
            time_start=time.time()
            for i in range(0, image_r.size(3)):   ## 0 --- (no_echos - 1)
                for j in range(0, image_r.size(2)):
                    INPUT_r = image_r[:,:,j,i]  ##  256 * 128
                    INPUT_i = image_i[:,:,j,i]

                    ###print(INPUT_r.size())

                    INPUT_r = torch.unsqueeze(INPUT_r, 0) ##  1 * 256 * 256
                    INPUT_r = torch.unsqueeze(INPUT_r, 0) ## 1 * 1 * 256 * 256

                    INPUT_i = torch.unsqueeze(INPUT_i, 0)  ## 1 * 256 * 256
                    INPUT_i = torch.unsqueeze(INPUT_i, 0)  ## 1 * 1 * 256 * 256
                    ## ready for reconstruction. 
                ################ Evaluation ##################
                    INPUT_r = INPUT_r.to(device)
                    INPUT_i = INPUT_i.to(device)

                    ## 
                    INPUT_k_r = k0_r[:,:,j,i]  ##  256 * 128
                    INPUT_k_i = k0_i[:,:,j,i]

                    ###print(INPUT_k_r.size())

                    INPUT_k_r = torch.unsqueeze(INPUT_k_r, 0) ##  1 * 256 * 256
                    INPUT_k_r = torch.unsqueeze(INPUT_k_r, 0) ## 1 * 1 * 256 * 256

                    INPUT_k_i = torch.unsqueeze(INPUT_k_i, 0)  ## 1 * 256 * 256
                    INPUT_k_i = torch.unsqueeze(INPUT_k_i, 0)  ## 1 * 1 * 256 * 256
                    ## ready for reconstruction. 
                ################ Evaluation ##################
                    INPUT_k_r = INPUT_k_r.to(device)
                    INPUT_k_i = INPUT_k_i.to(device)

                    pred_r, pred_i = octnet(INPUT_r, INPUT_i, INPUT_k_r, INPUT_k_i, mask)

                    pred_r = torch.squeeze(pred_r, 0)  ## 1 * 256 * 256
                    pred_i = torch.squeeze(pred_i, 0)  ## 1 * 256 * 256

                    pred_r = torch.squeeze(pred_r, 0)  ##  256 * 256
                    pred_i = torch.squeeze(pred_i, 0)  ##  256 * 256
                    
                    recons_r[:,:,j,i] = pred_r
                    recons_i[:,:,j,i] = pred_i
            time_end =time.time()
            print(time_end - time_start)

            recons_r = recons_r.to('cpu')
            recons_r = recons_r.numpy()
            recons_i = recons_i.to('cpu')
            recons_i = recons_i.numpy()

            print('Saving results')
            path = "recon_real_" + datafiles["name"][12:]
            scio.savemat(path, {'recons_r':recons_r})

            path = "recon_imag_" + datafiles["name"][12:]
            scio.savemat(path, {'recons_i':recons_i})
            print('end')