CGMSeriesLunchPat1 = readtable('CGMSeriesLunchPat1.csv');
CGMSeriesLunchPat2 = readtable('CGMSeriesLunchPat2.csv');
CGMSeriesLunchPat3 = readtable('CGMSeriesLunchPat3.csv');
CGMSeriesLunchPat4 = readtable('CGMSeriesLunchPat4.csv');

CGMDatenumLunchPat1 = readtable('CGMDatenumLunchPat1.csv');
CGMDatenumLunchPat2 = readtable('CGMDatenumLunchPat2.csv');
CGMDatenumLunchPat3 = readtable('CGMDatenumLunchPat3.csv');
CGMDatenumLunchPat4 = readtable('CGMDatenumLunchPat4.csv');

InsulinBasalLunchPat1 = readtable('InsulinBasalLunchPat1.csv');
InsulinBolusLunchPat1 = readtable('InsulinBolusLunchPat1.csv');
InsulinDatenumLunchPat1 = readtable('InsulinDatenumLunchPat1.csv');



%Fst fourier transform calculation
%{
hold on
axl = nexttile;
plot(axl,CGMDatenumLunchPat4{1,:}, CGMSeriesLunchPat4{1,:});
hold off

Y = fft(CGMSeriesLunchPat4{1,:});
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;

hold on
axl = nexttile;
plot(axl,f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
hold off
%}

for c = 1:height(CGMDatenumLunchPat1)

    for i=1:length(CGMSeriesLunchPat1{c,:})    
        if isnan(CGMSeriesLunchPat1{c,i}) %Handle Missing values
              CGMSeriesLunchPat1{c,i}=0;
        end 
    end
    
    Fs = length(CGMDatenumLunchPat1{c,:})/9000;            % Sampling frequency                    
    T = 1/Fs;                                           % Sampling period       
    L = length(CGMDatenumLunchPat1{c,:});             % Length of signal
    t = (0:L-1)*T;                                  % Time vector
    
    Y = fft(CGMSeriesLunchPat1{c,:});
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:(L/2))/L;

    hold on
    axl = nexttile;
    plot(axl,f,P1) 
    title('Single-Sided Amplitude of X(t)')
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    hold off
    
end





