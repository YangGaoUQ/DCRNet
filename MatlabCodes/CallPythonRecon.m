
%% Call Python script to conduct the reconstruction; 
curDir = pwd; 

if ispc
    ConfigPython; 
end

disp('Calling Python for DCRNet-based MRI reconstruction'); 

cd ../PythonCodes/
!python -u ../PythonCodes/Inference.py
cd(curDir)

disp('DCRNet-based Reconstruction Finished');