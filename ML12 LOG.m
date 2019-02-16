close all
clc

HouseParamTrain=dataset(1:1100,4:8); %rename the imported file to dataset
HouseParamCV=dataset(1101:1800,4:8);

PriceTrain=dataset(1:1100,1); 
mTrain=length(PriceTrain);

PriceCV=dataset(1101:1800,1); 
mCV=length(PriceCV);

NumOfParam=5;
alpha=0.001;
iter=500;
mseTrain=[];
mseCV=[];
costTrain=[];
costCV=[];
iterr=[];


for i=1:NumOfParam
    
HouseParamTrain(:,i)=(HouseParamTrain(:,i)-mean(HouseParamTrain(:,i)))/std(HouseParamTrain(:,i));

HouseParamCV(:,i)=(HouseParamCV(:,i)-mean(HouseParamCV(:,i)))/std(HouseParamCV(:,i));
end

HouseParamTrain=[ones(mTrain,1) HouseParamTrain HouseParamTrain.^2];

HouseParamCV=[ones(mCV,1) HouseParamCV HouseParamCV.^2];
 
 
PriceTrain=(PriceTrain-mean(PriceTrain))/std(PriceTrain);
 
PriceCV=(PriceCV-mean(PriceCV))/std(PriceCV);

thetaTrain=randn(size(HouseParamTrain,2),1);
thetaCV=randn(size(HouseParamCV,2),1);

 for i=1:iter
     
    HypoTrain=Hypoth(HouseParamTrain,thetaTrain);
    HypoCV=Hypoth(HouseParamCV,thetaCV);
    
    costTrain= (1/(2*mTrain)).*sum((HypoTrain-PriceTrain).^2);
    costCV= (1/(2*mCV)).*sum((HypoCV-PriceCV).^2);

    mseTrain=[mseTrain;costTrain];
    mseCV=[mseCV;costCV];
    
    iterr=[iterr;i];
    
    newThetaTrain=GradDesc(HouseParamTrain,HypoTrain,PriceTrain,mTrain,thetaTrain,alpha,size(HouseParamTrain,2));
    newThetaCV=GradDesc(HouseParamCV,HypoCV,PriceCV,mCV,thetaCV,alpha,size(HouseParamCV,2));
    
    thetaTrain=newThetaTrain;
    thetaCV=newThetaCV;
end
 
figure (1)
plot(iterr,mseCV,'b')
hold on
plot(iterr,mseTrain,'g')
legend('Cross Validation','Training Set')
title('House Price')
xlabel('Iterations')
ylabel('Cost Function')