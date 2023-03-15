function [recall_rate, prec_rate, f1score] = evaluate_perform(true_labels, predict_labels)

true_labels    = (true_labels > 0); 
predict_labels = (predict_labels > 0); 

% recall rate: true positive/ # total ground truth positive
recall_rate = sum(true_labels.*predict_labels)/sum(true_labels); 

% precision rate: true positive/# total predicted positive
prec_rate   = sum(true_labels.*predict_labels)/sum(predict_labels);

f1score = 2 * (recall_rate * prec_rate) /(recall_rate + prec_rate); 

end 
