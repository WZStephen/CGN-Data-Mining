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



for c = 1:height(CGMDatenumLunchPat1)
    
    %Find the max and min of glucose level from each day(row)
    [val,loc] = max(CGMSeriesLunchPat1{c,:});
    CGM1MaxTable(c,1) = val;
    CGM1MaxTable(c,2) = table2array(CGMDatenumLunchPat1(c,loc));
    
    [val,loc] = min(CGMSeriesLunchPat1{c,:});
    CGM1MinTable(c,1) = val;
    CGM1MinTable(c,2) = table2array(CGMDatenumLunchPat1(c,loc));
    
    %Find the mean and variance of glucose level from each day(row)
    %If there exit NaN, which means the data set in that row is not
    %complete
    sum=0;
    for i=1:length(CGMSeriesLunchPat1{c,:})
        
        if isnan(CGMSeriesLunchPat1{c,i}) %Handle Missing values
          CGMSeriesLunchPat1{c,i}=0;
        end
        
        sum=sum+CGMSeriesLunchPat1{c,i};
    end
    Mean=sum/length(CGMSeriesLunchPat1{c,:}); %Mean
    
    sum1=0;
    for i=1:length(CGMSeriesLunchPat1{c,:})
        
        if isnan(CGMSeriesLunchPat1{c,i}) %Handle Missing values
              CGMSeriesLunchPat1{c,i}=0;
        end
        
        sum1=sum1+(CGMSeriesLunchPat1{c,i}-Mean)^2;
    end
    Variance=sum1/length(CGMSeriesLunchPat1{c,:}); %Varaince
    
    CGM1MeanAndVarianceTable(c,1) = Mean; %Store the Mean to table
    CGM1MeanAndVarianceTable(c,2) = Variance; %Store the Variance to table
    CGM1MeanAndVarianceTable(c,3) = sqrt(Variance); %Store the Standard Deviation to table
    
    %Convert datenum to actual date
    t = CGMDatenumLunchPat1{c,:};
    t1 = datetime(t(1:end), 'ConvertFrom','datenum'); 
    
    %plot graph
    hold on 
    axl = nexttile;
    plot(axl,t1,CGMSeriesLunchPat1{c,:});
    yline(Mean);
end
hold off

SumOfMean = 0;
for c=1:33
    SumOfMean = SumOfMean + CGM1MeanAndVarianceTable(c,1);
end
AverageOfMean = SumOfMean/length(CGM1MeanAndVarianceTable);

hold on
axl = nexttile;
plot(axl, CGM1MeanAndVarianceTable(:,1));
axl = nexttile;

plot(axl, CGM1MeanAndVarianceTable(:,2));
axl = nexttile;

plot(axl, CGM1MeanAndVarianceTable(:,3));

hold off

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

