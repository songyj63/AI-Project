
load('train_set.mat')

cnt = 1;
for i=1:20%size(train_set,1)
    path = strcat('Features(covarep_formant)/', int2str(train_set(i,1)), '_P_features.mat');
    tmpData = load(path);
    tmpData = tmpData.features;
    
    for j=1:size(tmpData,1)/300
        X_training(cnt,:,:) = tmpData((j-1)*300+1 : j*300,:);
        labelSet_training(cnt,:) = train_set(i,:);
        cnt = cnt + 1;
    end
    
    i
end

%%

x1 = load('labelSet_training_1_56.mat');
x2 = load('labelSet_training_57_70.mat');
x3 = load('labelSet_training_71_90.mat');
x4 = load('labelSet_training_91_110.mat');
x5 = load('labelSet_training_111_125.mat');
x6 = load('labelSet_training_126_138.mat');

y1 = load('X_training_1_56.mat');
y2 = load('X_training_57_70.mat');
y3 = load('X_training_71_90.mat');
y4 = load('X_training_91_110.mat');
y5 = load('X_training_111_125.mat');
y6 = load('X_training_126_138.mat');

labelSet_training = [x1.labelSet_training; x2.labelSet_training; x3.labelSet_training; x4.labelSet_training; x5.labelSet_training; x6.labelSet_training];
X_training = [y1.X_training; y2.X_training; y3.X_training; y4.X_training; y5.X_training; y6.X_training];



