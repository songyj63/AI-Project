
% taking audio file from only participant (excluding robot interviewer)

user = 300;
path = strcat(int2str(user), '_P/');
transcript_fileName = strcat(path, int2str(user), '_TRANSCRIPT.csv');
audio_fileName = strcat(path, int2str(user), '_AUDIO.wav');
new_audio_fileName = strcat(path, int2str(user), '_AUDIO_Modified.wav');
trs_obj = tdfread(transcript_fileName);
% (start_time, stop_time, speaker, value)

[y, Fs] = audioread(audio_fileName);

convSize = size(trs_obj.speaker,1);
cnt = 0;
for i = 1:convSize
    
    if(trs_obj.speaker(i,1:end) == 'Participant')
        
        if(cnt == 0)
            yFromP = y(Fs*trs_obj.start_time(i):Fs*trs_obj.stop_time(i));
            cnt = 1;
        else
            yFromP = [yFromP; y(Fs*trs_obj.start_time(i):Fs*trs_obj.stop_time(i))];    
        end
        
    end
end

audiowrite(new_audio_fileName, yFromP, Fs);
