function OpenPyFolder(Dir)
%OPENPYFOLDER Summary of this function goes here
%   Detailed explanation goes here
if isunix
    system(['xdg-open ', Dir])
elseif ispc
    winopen(Dir)
elseif ismac
    system(['open ', Dir]);
end

end

