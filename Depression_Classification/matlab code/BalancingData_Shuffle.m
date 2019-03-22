% load('labelSet_training.mat')
% load('X_training_Normalised.mat')

idx = find(labelSet_training(:,2) == 1);

x = labelSet_training(idx,:);
x_t = X_training_Normalised(idx,:,:);

labelSet_training_balanced = [labelSet_training ; x];
X_training_Normalised_balanced = cat(1,X_training_Normalised,x_t);

randval = randperm(size(labelSet_training_balanced,1));
labelSet_training_balanced_shuffled = labelSet_training_balanced(randval,:);
X_training_Normalised_balanced_shuffled = X_training_Normalised_balanced(randval,:,:);