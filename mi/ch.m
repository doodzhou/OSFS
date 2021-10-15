function [ output ] = ch(X,Y)
%function output = ch(X,Y)
%X can be a matrix which is converted into a joint variable before calculation
%expects variables to be column-wise
%
%returns the entropy of X conditioned on y, H(X|Y)

if (size(X,2)>1)
	mergedVector = MIToolboxMex(3,X);
else
	mergedVector = X;
end
[output] = MIToolboxMex(6,mergedVector,Y);

