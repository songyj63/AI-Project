
% modified wav file -> 3sec multiple files

num = 422:492;
missing = [318 321 341 362 342 394 398 460];
N = setdiff(num, missing);

for i=N
    
    user = i;
    path = strcat('audio_Modified/');
    audio_fileName = strcat(path, int2str(user), '_AUDIO_Modified.wav');

    pathNew = strcat('audio_Modified_3s/', int2str(user), '_P/');

    [y, Fs] = audioread(audio_fileName);

    cnt = 0;
    sigLength = length(y);
    timeLength = Fs*3;   % 3s

    mkdir(pathNew);

    while( (cnt+1)*timeLength < sigLength)

        audiowrite(strcat(pathNew, int2str(user), '_AUDIO_Modified_', int2str(cnt), '.wav'), y((timeLength*cnt)+1:(cnt+1)*timeLength), Fs);
        cnt = cnt+1;

    end

    i
end