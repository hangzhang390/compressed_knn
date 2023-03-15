close all;
clear all;
clc; 

k = 20;

addpath('./cifar10/');   

load('cifar10_embedding_train.mat'); 
trainX = data(:, 1:512);
trainY = data(:, 513); 

load('cifar10_embedding_test.mat'); 
testX = data(:, 1:512);
testY = data(:, 513);
 
%% 
[~, p] = size(testX);

dp_ratio = 0.2:0.1:0.8;  

r_arr = zeros(10, 1); 
p_arr = zeros(10, 1); 
f_arr = zeros(10, 1); 

r_proj_mat = zeros(10, length(dp_ratio)); 
p_proj_mat = zeros(10, length(dp_ratio));
f_proj_mat = zeros(10, length(dp_ratio));

for target_digit = 0:1:9
    
    trainY_bin = change_labels(trainY, target_digit);
    testY_bin  = change_labels(testY,  target_digit);
    
    mdl            = fitcknn(double(trainX), trainY_bin, 'NumNeighbors', k, 'Distance', 'euclidean'); 
    predict_labels = predict(mdl, double(testX)).'; 

    % Evaluate the performance 
    [rval, pval, fval] = evaluate_perform(predict_labels, testY_bin); 

    r_arr(target_digit + 1) = rval; 
    p_arr(target_digit + 1) = pval; 
    f_arr(target_digit + 1) = fval;

    for dp_idx = 1:1:length(dp_ratio)
    
    d = round(p * dp_ratio(dp_idx)); 
    
    A = normrnd(0, 1, [d, p])/sqrt(d);

    % projection 
    testX_tilde  = double(testX)  * A.'; 
    trainX_tilde = double(trainX) * A.';
                       
    mdl_proj    = fitcknn(double(trainX_tilde), trainY_bin, 'NumNeighbors', k, 'Distance', 'euclidean'); 
    predict_proj_labels = predict(mdl_proj, double(testX_tilde)).'; 

    % Evaluate the performance 
   [r_proj, p_proj, f_proj] = evaluate_perform(predict_proj_labels, testY_bin); 

   r_proj_mat(target_digit + 1, dp_idx) = r_proj; 
   p_proj_mat(target_digit + 1, dp_idx) = p_proj; 
   f_proj_mat(target_digit + 1, dp_idx) = f_proj; 

   end
end 

save(['cifar10_embeddingwhole_' num2str(k) 'nn_results.mat'], 'r_arr', 'p_arr', 'f_arr', ...
                      'r_proj_mat', 'p_proj_mat', 'f_proj_mat');
   

                  
   


















