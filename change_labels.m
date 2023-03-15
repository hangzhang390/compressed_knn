function [labels_bin] = change_labels(labels, tar_label)

% Since our alogrithm only considers the 
% The target label is set as one; 
% The rest of labels is set as minus one; 

labels_bin = zeros(1, length(labels)); 

tar_pos    = find(labels == tar_label); 
nontar_pos = find(labels ~= tar_label); 

labels_bin(tar_pos)    = 1; 
labels_bin(nontar_pos) = -1; 

end