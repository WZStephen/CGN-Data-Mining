% Requirements: MealData.csv and Nomeal.csv for 5 patients

%% Reading Data

% Reading meal data
MealData1 = readmatrix('mealData1.csv');
MealData2 = readmatrix('mealData2.csv');
MealData3 = readmatrix('mealData3.csv');
MealData4 = readmatrix('mealData4.csv');
MealData5 = readmatrix('mealData5.csv');

MealData=[MealData1;MealData2;MealData3;MealData4;MealData5];

% Replacing all rows that are totally NaN
MealData(~any(~isnan(MealData),2),:) = [];

% Replacing NaN with medain of 5 values
MealData = fillmissing(MealData,'linear',2,'EndValues','nearest');

% Reading NoMeal data
NoMealData1 = readmatrix('Nomeal1.csv');
NoMealData2 = readmatrix('Nomeal2.csv');
NoMealData3 = readmatrix('Nomeal3.csv');
NoMealData4 = readmatrix('Nomeal4.csv');
NoMealData5 = readmatrix('Nomeal5.csv');

NoMealData = [NoMealData1;NoMealData2;NoMealData3;NoMealData4;NoMealData5];
%Replacing all rows that contain only NaN
NoMealData(~any(~isnan(NoMealData),2),:) = [];

% Replacing NaN with medain of 5 values
NoMealData = fillmissing(NoMealData,'linear',2,'EndValues','nearest');

%% Statistical Computation of Meal Data

CGM1Mean = nanmean(MealData(:,:),2);
CGM1Variance = nanvar(MealData(:,:),0,2);

CGM1Max = max(MealData(:,:),[],2);
CGM1Min = min(MealData(:,:),[],2);
CGM1Range = CGM1Max - CGM1Min;
CGM1Median = nanmedian(MealData(:,:),2);

CGMStatisticalMealFeatures = [CGM1Mean CGM1Variance CGM1Max CGM1Min CGM1Range CGM1Median];

%% Statistical Computing No Meal Data

CGM1MeanNoMeal = nanmean(NoMealData(:,:),2);
CGM1VarianceNoMeal = nanvar(NoMealData(:,:),0,2);

CGM1MaxNoMeal = max(NoMealData(:,:),[],2);
CGM1MinNoMeal = min(NoMealData(:,:),[],2);
CGM1RangeNoMeal = CGM1MaxNoMeal - CGM1MinNoMeal;
CGM1MedianNoMeal = nanmedian(NoMealData(:,:),2);

CGMStatisticalNoMealFeatures = [CGM1MeanNoMeal CGM1VarianceNoMeal CGM1MaxNoMeal CGM1MinNoMeal CGM1RangeNoMeal CGM1MedianNoMeal];

%% PolyFit Meal Data

[rowsMeal, colsMeal] = size(MealData);

for rowindex = 1:rowsMeal
    N = 5;
    polyCoeffMeal(rowindex,:) = polyfit(0.0034*(MealData(rowindex,1:colsMeal)),flip(MealData(rowindex,1:colsMeal)),N);
end
%% PolyFit No Meal Data

[rowsNoMeal, colsNoMeal] = size(NoMealData);
for rowindex = 1:rowsNoMeal
    N = 5;
    polyCoeffNoMeal(rowindex,:) = polyfit(0.0034*(NoMealData(rowindex,1:colsNoMeal)),flip(NoMealData(rowindex,1:colsNoMeal)),N);
end
%% FFT on Meal data

for c = 1:rowsMeal
    fftMeal(c,:) = abs(fft(MealData(c,:)));
end

%Taking only releevant 8 columns of fft ignoring the 1st column
top8fftMeal = fftMeal(:,2:9);
%% FFT on NoMeal Data

for c = 1:rowsNoMeal
    fftNoMeal(c,:) = abs(fft(NoMealData(c,:)));
end

%Taking only releevant 8 columns of fft ignoring the 1st column
top8fftNoMeal = fftNoMeal(:,2:9);
%% Basic Spectral Analysis Meal Data

[rows, cols] = size(MealData);
for c=1:rows
    x = MealData(c,1:30);          
    fs = length(x)/(150*60);       
    t = 0:1/fs:10-1/fs;            
    
    % computing discrete Fourier transform of the signal.
    y = fft(x);
    
    n = length(x);          
    f = (0:n-1)*(fs/n);     
    power = abs(y).^2/n;    
    
    bsaFeatureMeal(c,:) = power(1,4:11); 

end

%% Basic Spectral Analysis No Meal Data

[rowsNoMeal, colsNoMeal] = size(NoMealData);
for c=1:rowsNoMeal
    xNoMeal = NoMealData(c,:);
    fsNoMeal = length(xNoMeal)/(150*60);
    tNoMeal = 0:1/fsNoMeal:10-1/fsNoMeal;
    
    yNoMeal = fft(xNoMeal);
    
    nS = length(xNoMeal);
    fNoMeal = (0:nS-1)*(fsNoMeal/nS);
    powerNoMeal = abs(yNoMeal).^2/nS;
    
    bsaFeatureNoMeal(c,:) = powerNoMeal(1,4:11);

end
%% Concatenate the Feature type result into a Feature Matrix Meal Data

FeatureMatrixMeal = [CGMStatisticalMealFeatures polyCoeffMeal bsaFeatureMeal top8fftMeal];
%% Concatenate the Feature type result into a Feature Matrix No Meal Data

FeatureMatrixNoMeal = [CGMStatisticalNoMealFeatures polyCoeffNoMeal bsaFeatureNoMeal top8fftNoMeal];
%% Combined feature matrix

CombinedFeatureMatrix = [FeatureMatrixMeal; FeatureMatrixNoMeal];

%% Normalize the combined Feature Matrix 

FeatureMatrixCombined_norm = normalize(CombinedFeatureMatrix,'range');

%% Perform PCA to pick top 5 feature types
[coeff, score, latent, tsquared, explained] = pca(FeatureMatrixCombined_norm);

EigenVectorsTop5 = coeff(:, 1:5);
csvwrite('EigenVectorsTop5.csv', EigenVectorsTop5);
NewFeatureMatrix = FeatureMatrixCombined_norm * EigenVectorsTop5;
CombinedFeatures = NewFeatureMatrix;

%% Segregating Meal and No meal dataset and adding label column at the end
MealFeatures = CombinedFeatures(1:rowsMeal,:);

vec=zeros(rowsMeal,1);
vec(:)=1;

MealFeatures = [MealFeatures vec];

NoMealFeatures = CombinedFeatures(rowsMeal+1:rowsMeal+rowsNoMeal,:);

vec=zeros(rowsNoMeal,1);

NoMealFeatures = [NoMealFeatures vec];
%% Splitting training and testing data

[m,n] = size(MealFeatures) ;
P = 0.8;
idx = randperm(m)  ;

TrainingMealData = MealFeatures(idx(1:round(P*m)),:) ;
TestingMealData = MealFeatures(idx(round(P*m)+1:end),:) ;

[m,n] = size(NoMealFeatures) ;
P = 0.8 ;
idx = randperm(m)  ;
TrainingNoMealData = NoMealFeatures (idx(1:round(P*m)),:) ;
TestingNoMealData = NoMealFeatures (idx(round(P*m)+1:end),:) ;

TotalTrainingData = [TrainingMealData ; TrainingNoMealData];
TotalTestData = [TestingMealData;TestingNoMealData];

shuffledTrainData =  TotalTrainingData(randperm(end),:);
shuffledTestData =  TotalTestData(randperm(end),:);

%% writing training and testing data in files 

% csvwrite('trainData.csv',shuffledTrainData);
% csvwrite('testData.csv',shuffledTestData);
%% Training models

trainFeatures = shuffledTrainData(:,1:5);
trainLabels = shuffledTrainData(:,6);

testFeatures = shuffledTestData(:,1:5);
testLabels = shuffledTestData(:,6);

%% SVM
SVM = fitcsvm(trainFeatures,trainLabels);
SVMModel =  crossval(SVM);
saveLearnerForCoder(SVMModel.Trained{1},'SVM');
[SVMlabels,score] = predict(SVMModel.Trained{1},testFeatures);
[SVMp,SVMr,SVMa,SVMf1] = Evaluate(testLabels, SVMlabels);

%% KNN
KNN = fitcknn(trainFeatures,trainLabels, 'NumNeighbors', 5, 'Standardize', 1);
KNNModel = crossval(KNN);
saveLearnerForCoder(KNNModel.Trained{1}, 'KNN');
[label,score] = predict(KNNModel.Trained{1},testFeatures);
[KNNp, KNNr, KNNa, KNNf1] = Evaluate(testLabels, label);

%% naive bayes
NB = fitcnb(trainFeatures,trainLabels);
NBModel = crossval(NB);
saveLearnerForCoder(NBModel.Trained{1}, 'NaiveBays');
[label,score] = predict(NBModel.Trained{1},testFeatures);
[NBp, NBr, NBa, NBf1] = Evaluate(testLabels, label);

%% Decision Tree
DT = fitctree(trainFeatures,trainLabels);
DTModel = crossval(DT);
saveLearnerForCoder(DTModel.Trained{1}, 'DecisionTree');
[label,score] = predict(DTModel.Trained{1},testFeatures);
[DTp, DTr, DTa, DTf1] = Evaluate(testLabels, label);

%% clear all useless variables
clear MealData5 MealData4 MealData3 MealData2 MealData1;
clear vec
clear idx p n N tp tn fp fn;
clear NoMealData1 NoMealData2 NoMealData3 NoMealData4 NoMealData5
clear CGM1Median CGM1Mean CGM1Max CGM1Min CGM1Range CGM1Variance;
clear CGM1VarianceNoMeal CGM1MaxNoMeal CGM1MedianNoMeal CGM1MinNoMeal CGM1RangeNoMeal CGM1MeanNoMeal;
clear rowindex N;
clear fftMeal;
clear fftNoMeal c;
clear c x fs t y n f power;
clear CGMStatisticalMealFeatures polyCoeffMeal bsaFeatureMeal top8fftMeal;
clear CGMStatisticalNoMealFeatures polyCoeffNoMeal bsaFeatureNoMeal top8fftNoMeal;
clear coeff score latent tsquared explained NewFeatureMatrix EigenVectorsTop5 FeatureMatrixCombined_norm CombinedFeatureMatrix;
clear cols colsMeal colsNoMeal CombinedFeatures FeatureMatrixNoMeal FeatureMatrixMeal fNoMeal fsNoMeal idx label MealData MealFeatures;
clear n NoMealData NoMealFeatures nS P powerNoMeal rows rowsMeal rowsNoMeal score shuffledTestData shuffledTrainData testFeatures TestingMealData;
clear TestingNoMealData testLabels tNoMeal TotalTestData TotalTrainingData trainFeatures TrainingNoMealData TrainingMealData
clear train xNoMeal

%% functin to measure performance on test data

function [precision, recall, accuracy, F1] = Evaluate(testLabels, label)
    idx = (testLabels ()==1);
    p = length(testLabels (idx));
    n = length(testLabels (~idx));
    N = p+n;
    tp = sum(testLabels (idx)==label (idx));
    tn = sum(testLabels (~idx)==label (~idx));
    fp = n-tn;
    fn = p-tp;

    precision = tp/(tp + fp);
    recall = tp/(tp + fn);
    accuracy = (tp+ tn)/(tp+fn+tn+fp);
    F1 = 2*precision*recall/(precision + recall);
end
