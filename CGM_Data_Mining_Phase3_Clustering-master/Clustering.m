function output = Clutering(inputFile)
% class label 1 for meal data and 0 for no meal data
% sample input [173   173   172   173   165   161   159   154   148   145   144   144   147   146   144   144   143   140   139   136   135   134   134   134   136   136   137   138   138   139]

%MealData = readmatrix(inputFile);
MealData = readmatrix(inputFile);


[row,col] = size(MealData);

output = [];
%% Statistical Computing Meal Data
[rows, cols] = size(MealData);
CGM1Mean = nanmean(MealData(:,:),2);
CGM1Variance = nanvar(MealData(:,:),0,2);

CGM1Max = max(MealData(:,:),[],2);
CGM1Min = min(MealData(:,:),[],2);
CGM1Range = CGM1Max - CGM1Min;
CGM1Median = nanmedian(MealData(:,:),2);

CGMStatisticalFeatures = [CGM1Mean CGM1Variance CGM1Max CGM1Min CGM1Range CGM1Median];


%% PolyFit Meal Data
[rows, cols] = size(MealData);
for rowindex = 1:rows
    N = 5;
    polyCoeff(rowindex,:) = polyfit(0.0034*(MealData(rowindex,1:cols)),flip(MealData(rowindex,1:cols)),N);
end

%% FFT as in class Meal Data
for c = 1:rows
    fftMeal1(c,:) = abs(fft(MealData(c,:)));
end
%Taking only releevant 8 columns of fft ignoring the 1st column
top8fft = fftMeal1(:,2:9);

%% Linear Predictive Coding(LPC) Meal Data
%Linear prediction is a mathematical operation where future values of-
%a discrete-time signal are estimated as a linear function of previous samples.
[rows, cols] = size(MealData);
for c = 1:rows
%Define the signal data
x = MealData(c,1:30)';
%Compute the predictor coefficients.
% finds the coefficients of a pth-order linear predictor,
%an FIR filter that predicts the current value of the real-valued time series x based on past samples.
%The function also returns g, the variance of the prediction error.
%If x is a matrix, the function treats each column as an independent channel.
a = lpc(x, 3);
%Compute the estimated signal
est_x = filter([0 -a(2:end)],1,x);
lpcFeature(c,:) = est_x'; %concatenate feature with time series
end

%% Basic Spectral Analysis Meal Data
[rows, cols] = size(MealData);
for c=1:rows
x = MealData(c,1:30);%Define the signal data
fs = length(x)/(150*60);       %Sampling frequency (About Every 5 Minutes to gather one data)
t = 0:1/fs:10-1/fs;            % 10 second span time vector

%The Fourier transform of the signal identifies its frequency components.
%In MATLAB?, the fft function computes the Fourier transform using a fast Fourier transform algorithm.
%Use fft to compute the discrete Fourier transform of the signal.
y = fft(x);

n = length(x);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
power = abs(y).^2/n;    % power of the DFT

bsaFeature(c,:) = power(1,4:11); %concatenate feature with time series ignoring first 4 and last 4
end

%% Concatenate the Feature type result into a Feature Matrix Meal Data
FeatureMatrix = [CGMStatisticalFeatures polyCoeff bsaFeature top8fft];

%% Normalize the Feature Matrix Meal Data
FeatureMatrix_norm = normalize(FeatureMatrix,'range');

%% Perform PCA to pick top 5 feature types Meal Data
[coeff, score, latent, tsquared, explained] = pca(FeatureMatrix_norm);
EigenVectorsTop5 = coeff(:, 1:5);
NewFeatureMatrix = FeatureMatrix_norm * EigenVectorsTop5;
writematrix(NewFeatureMatrix, 'output_Top5Feature.csv');

%Implementing the k-mean and DBSCAN to do clusterting
[idx,C] = kmeans(NewFeatureMatrix,10);
[idx1,C] = dbscan(NewFeatureMatrix,0.2,2);
%Generate the Serial number to identification
for i=1 : 245
Serial_Num(i,1) = i;
end
%Combine and write the result
Cluster_Output = [Serial_Num MealData idx idx1];
writematrix(Cluster_Output, 'Cluster_Output.csv');
disp('Please check Cluster_output.csv file for results')
end