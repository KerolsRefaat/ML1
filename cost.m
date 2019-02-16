function [ cost ] = cost(hypo,Price,m)


cost= (1/(2*m))*sum((hypo-Price).^2);


end

