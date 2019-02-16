close all
clc

%rename the imported file to dataset
HouseParamTrain=dataset(1:1100,4:8); %BestOne
%HouseParam=dataset(:,9:13);
%HouseParam=dataset(:,14:18);
%HouseParam=dataset(:,7:12);
HouseParamCV=dataset(1101:1500,4:8);
HouseParamTest=dataset(1501:1800,4:8);

PriceTrain=dataset(1:1100,1); 
mTrain=length(PriceTrain);

PriceCV=dataset(1101:1500,1); 
mCV=length(PriceCV);

PriceTest=dataset(1501:1800,1); 
mTest=length(PriceTest);

NumOfParam=5;
alpha=0.001;
iter=1000;
mseTrain=[];
mseCV=[];
costTrain=[];
costCV=[];
mseTest=[];
costTest=[];
iterr=[];


for i=1:NumOfParam
    
HouseParamTrain(:,i)=(HouseParamTrain(:,i)-mean(HouseParamTrain(:,i)))/std(HouseParamTrain(:,i));

HouseParamCV(:,i)=(HouseParamCV(:,i)-mean(HouseParamCV(:,i)))/std(HouseParamCV(:,i));

HouseParamTest(:,i)=(HouseParamTest(:,i)-mean(HouseParamTest(:,i)))/std(HouseParamTest(:,i));
end

HouseParamTrain=[ones(mTrain,1) HouseParamTrain HouseParamTrain.^2];

HouseParamCV=[ones(mCV,1) HouseParamCV HouseParamCV.^2];
 
HouseParamTest=[ones(mTest,1) HouseParamTest HouseParamTest.^2];

% 4 diff hypoth
% HouseParam=[ones(length(HouseParam),1) HouseParam HouseParam.^2 HouseParam.^3];
% HouseParam=[ones(length(HouseParam),1) HouseParam 2.*HouseParam];
% HouseParam=[ones(length(HouseParam),1) HouseParam ];
 
PriceTrain=(PriceTrain-mean(PriceTrain))/std(PriceTrain);
PriceCV=(PriceCV-mean(PriceCV))/std(PriceCV);
PriceTest=(PriceTest-mean(PriceTest))/std(PriceTest);

thetaTrain=randn(size(HouseParamTrain,2),1);
thetaCV=randn(size(HouseParamCV,2),1);
thetaTest=randn(size(HouseParamTest,2),1);


 for i=1:iter
     
    HypoTrain=Hypoth(HouseParamTrain,thetaTrain);
    HypoCV=Hypoth(HouseParamCV,thetaCV);
    HypoTest=Hypoth(HouseParamTest,thetaTest);
    
    costTrain= (1/(2*mTrain)).*sum((HypoTrain-PriceTrain).^2);
    costCV= (1/(2*mCV)).*sum((HypoCV-PriceCV).^2);
    costTest= (1/(2*mTest)).*sum((HypoTest-PriceTest).^2);

    mseTrain=[mseTrain;costTrain];
    mseCV=[mseCV;costCV];
    mseTest=[mseTest;costTest];
    
    iterr=[iterr;i];
    
    newThetaTrain=GradDesc(HouseParamTrain,HypoTrain,PriceTrain,mTrain,thetaTrain,alpha,size(HouseParamTrain,2));
    newThetaCV=GradDesc(HouseParamCV,HypoCV,PriceCV,mCV,thetaCV,alpha,size(HouseParamCV,2));
    newThetaTest=GradDesc(HouseParamTest,HypoTest,PriceTest,mTest,thetaTest,alpha,size(HouseParamTest,2));
    
    thetaTrain=newThetaTrain;
    thetaCV=newThetaCV;
    thetaTest=newThetaTest;
end
 
figure (1)
plot(iterr,mseCV,'b')
hold on
plot(iterr,mseTrain,'g')
hold on
plot(iterr,mseTest,'r')
legend('Cross Validation','Training Set','Test')
title('House Price')
xlabel('Iterations')
ylabel('Cost Function')