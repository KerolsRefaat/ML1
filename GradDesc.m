function [ newTheta ] = GradDesc( houseP,hypo,Price,m,theta,alpha,NumOfParam)

newTheta=[];

for i=1:NumOfParam
    difference=(1/m)*sum((hypo-Price).*houseP(:,i));
    newtheta=theta(i,1)-(difference*alpha);
    newTheta=[newTheta;newtheta];  
end


end

