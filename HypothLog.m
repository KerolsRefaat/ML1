function [ hypo] = HypothLog( features,theta )
q=1+(exp(features.*theta));
hypo=1/q;


end

