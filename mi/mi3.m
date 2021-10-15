function [ output ] = mi3( X,Y,Z )
%MI3 MI(X,Y,Z)
% I(X;Y;Z)=I({X,Y};Z)-I(X;Z)-I(Y;Z)

% XY=[X,Y];
% output=mi(XY,Z)-mi(X,Z)-mi(Y,Z);

output=mi(X,Y)-cmi(X,Y,Z);
end

