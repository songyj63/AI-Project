
for i = 1:45

    num = i;
    
    path = 'Data/Dementia/';
    fileName = strcat('D (', num2str(i), ').txt');
    pathNew = 'Data/Data_FFT/Dementia/';
    fileNew = strcat('D (', num2str(i), ')_FFT.csv');
    
    xData = load(strcat(path, fileName));
    xData = xData(:,2:3);
    
    Fs = 256;
    Wpb = [3.5/(Fs/2) 50.5/(Fs/2)];                 % band-pass 4~50 Hz
    N = buttord(1/(Fs/2), 0.4/(Fs/2), 3, 30);       % passband ripple 3dB, stopband attenuation 30dB
    [Bpb,Apb] = butter(N,Wpb);
    % freqz(Bpb,Apb,256,256);
    % notch filter at 60Hz
    Wpn = [59/(Fs/2) 61/(Fs/2)];
    [Bpn,Apn] = butter(N, Wpn, 'stop');
    
    Filtered = Filtering_func_withNotchFilt(xData,Bpb,Apb,Bpn,Apn);
    
    %%
    
    % 4-8 Theta
    % 8-12 Alpha or Mu (if its near sensory-motor cortex)
    % 12-16 Low Beta
    % 16-20 Beta
    % 20-30 High Beta
    % 30-50 Gamma
    
    L = size(Filtered,1); % length of signal fft
    xaxisHz = Fs*(1:(L/2))/L;
    
    for ch=1:2
        X = Filtered(:,ch);
        Y = fft(X);
        Pow(:,ch) = (2*abs(Y(2:L/2+1))/L).^2;   % FFT is symmetric, throw away second half
        % Take the magnitude of fft of x, *2 because we made it half
        % return voltage value -> y-axis is same as time domain one
        % square power
    end
    
    idx = find(xaxisHz <= 50); % taking from 4Hz to 50Hz
    PowNew = Pow(idx,:);
    
    csvwrite(strcat(pathNew, fileNew), PowNew);
    clear;
end
