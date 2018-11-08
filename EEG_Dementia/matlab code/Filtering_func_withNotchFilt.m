function return_sig = Filtering_func_withNotchFilt(sig,Bpb,Apb,Bpn,Apn)

if(size(sig,2) == 1) %if it is one dimentional (EOG, Reference)
    %disp('1 dimentional Filter')
    %Band-pass EEG
    Filtered_sig = filtfilt(Bpb,Apb,sig);
    %notch filter at 50Hz
    Filtered_sig = filtfilt(Bpn,Apn,Filtered_sig);
    %set means to zero
    mean_sig = mean(Filtered_sig);
    return_sig = Filtered_sig - mean_sig;
else % two dimentional (EEG)
    %disp('2 dimentional Filter')
    for z=1:size(sig,2)
        %Band-pass EEG
        Filtered_sig(:,z) = filtfilt(Bpb,Apb,sig(:,z));
        %notch filter at 60Hz
        Filtered_sig(:,z) = filtfilt(Bpn,Apn,Filtered_sig(:,z));
        %set means to zero
        mean_sig = mean(Filtered_sig(:,z));
        Filtered_sig(:,z) = Filtered_sig(:,z)-mean_sig;
    end
    return_sig = Filtered_sig;
end


