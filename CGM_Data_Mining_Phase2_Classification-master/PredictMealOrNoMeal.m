function output = PredictMealOrNoMeal(inputFile)
% class label 1 for meal data and 0 for no meal data
% sample input [173   173   172   173   165   161   159   154   148   145   144   144   147   146   144   144   143   140   139   136   135   134   134   134   136   136   137   138   138   139]

lineInputs = readmatrix(inputFile);
[row,col] = size(lineInputs);
%% Read eigenvectors
EigenVectorsTop5 = readmatrix("EigenVectorsTop5.csv");
output = [];
for i = 1 : row
%% replace missing value
input = fillmissing(lineInputs(i,:),'linear',2,'EndValues','nearest');

%% Statistical Computation of Meal Data

CGM1Mean = nanmean(input(:),1);
CGM1Variance = nanvar(input(:),0,1);

CGM1Max = max(input(:),[],1);
CGM1Min = min(input(:),[],1);
CGM1Range = CGM1Max - CGM1Min;
CGM1Median = nanmedian(input(:),1);

CGMStatisticalMealFeatures = [CGM1Mean CGM1Variance CGM1Max CGM1Min CGM1Range CGM1Median];

%% PolyFit Meal Data

N = 5;
polyCoeffMeal(:) = polyfit(0.0034*(input(:)),flip(input(:)),N);

%% FFT on Meal data

fftMeal(:) = abs(fft(input(:)));

%Taking only releevant 8 columns of fft ignoring the 1st column
top8fftMeal = fftMeal(1,2:9);


%% Basic Spectral Analysis Meal Data

x = input(1,1:30);          
fs = length(x)/(150*60);       
t = 0:1/fs:10-1/fs;            

% computing discrete Fourier transform of the signal.
y = fft(x);

n = length(x);          
f = (0:n-1)*(fs/n);     
power = abs(y).^2/n;    

bsaFeatureMeal(:) = power(1,4:11); 


FeatureMatrix = [CGMStatisticalMealFeatures polyCoeffMeal bsaFeatureMeal top8fftMeal];

FeatureMatrix_norm = normalize(FeatureMatrix,'range');

NewFeatureMatrix = FeatureMatrix_norm * EigenVectorsTop5;
SVMModel = loadLearnerForCoder('SVM');
[classLabel_SVM,score] = predict(SVMModel,NewFeatureMatrix);

KNNModel = loadLearnerForCoder('KNN'); 
[classLabel_KNN,score] = predict(KNNModel,NewFeatureMatrix);

NBModel = loadLearnerForCoder('NaiveBays'); 
[classLabel_NB,score] = predict(NBModel,NewFeatureMatrix);

DTModel = loadLearnerForCoder('DecisionTree'); 
[classLabel_DT,score] = predict(DTModel,NewFeatureMatrix);

    output(i,1) = classLabel_SVM;
    output(i,2) = classLabel_KNN;
    output(i,3) = classLabel_NB;
    output(i,4) = classLabel_DT;
end
output = array2table(output, 'VariableNames', {'SVM', 'KNN', 'NaiveBays', 'DecisionTree'});
writetable(output, 'output.csv');


end