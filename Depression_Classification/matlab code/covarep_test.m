
% feature extraction 

num = 300:492;
missing = [318 321 341 362 342 394 398 460];
N = setdiff(num, missing);

for i=N

    user = int2str(i);
    dirPath= strcat('C:\Users\Administrator\Desktop\Programming\Matlab\AI\Depression\audio_Modified_3s\', user, '_P');
    %COVAREP_feature_extraction(dir,0.01)
    COVAREP_formant_extraction(dirPath,0.01)

    strcat('finish',user)

end

%%

% feature summation 

num = 300:492;
missing = [318 321 341 362 342 394 398 460];
N = setdiff(num, missing);

for i = N
    
    user = int2str(i);
    dirPath= strcat('C:\Users\Administrator\Desktop\Programming\Matlab\AI\Depression\audio_Modified_3s\', user, '_P\');

    d = dir([dirPath '\*.mat']);
    n = length(d);

    cnt = 0;
    fileName = strcat(user, '_AUDIO_Modified_', int2str(cnt), '.mat');
    x = load(strcat(dirPath, fileName));
    formantsWhole = x.formants;


    while(cnt < n-1)

        cnt = cnt+1;
        fileName = strcat(user, '_AUDIO_Modified_', int2str(cnt), '.mat');
        x = load(strcat(dirPath, fileName));
        formantsWhole = [formantsWhole ; x.formants];

    end

    save(strcat('Formants/',user, '_P_formants.mat'), 'formantsWhole');
    
    i
    
end

%%

% discard feature (recording silent error)

path = 'C:\Users\Administrator\Desktop\Programming\Matlab\AI\Depression\Feature(covarep_formant)\';
pathN = 'C:\Users\Administrator\Desktop\Programming\Matlab\AI\Depression\Features(covarep_formant)\';

num = 300:492;
missing = [318 321 341 362 342 394 398 460];
N = setdiff(num, missing);

for i=N

    user = int2str(i);
    fileName = strcat(user, '_P_features.mat');
    load(strcat(path, fileName));
    
    idx = 1;

    while(idx*300 <= size(features, 1))
        temp = features((idx-1)*300+1:idx*300, 75);

        if(sum(temp(:)==0) > 100)   % 1초 이상 무음
            features((idx-1)*300+1:idx*300, :) = [];
            (idx-1)*300+1;
        else
            idx = idx+1;
        end
    end    

    save(strcat(pathN, user, '_P_features.mat'), 'features');
    i
end
    
