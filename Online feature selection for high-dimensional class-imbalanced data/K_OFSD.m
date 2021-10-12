function [ selectedFeatures,time ] = K_OFSD( X, Y,N)
% Peng Zhou; Xuegang Hu; Peipei Li; Xindong Wu
% Online feature selection for high-dimensional class-imbalanced data
% Knowledge-Based Systems 2017 Vol.136
%   
% Output:  selectedFeatures  the index of selected features
%                time    running time
% Input:  X     data samples vector
%            Y     label vector
%            N     the parameter K
%   

%计算运行时间
start=tic;
[~,p]=size(X);

mode=zeros(1,p);                                             
dep_last=0;                                               

depArray=zeros(1,p);                                       

alpha=0.5;

for i=1:p
    col=X(:,i);
    dep_single=dep_n1(col,Y,N);                  
    depArray(1,i)=dep_single;
    
    if dep_single>alpha                                     
        
        if dep_single>dep_last
            mode((mode==1))=0;
            mode(1,i)=1;
            dep_last=dep_single;                                 
        else 
           mode(1,i)=1;             
           cols=X(:,mode==1);                                   
           dep_all=dep_n1(cols,Y,N);
           if dep_all>dep_last                                
                dep_last=dep_all;
           else
                 mode(1,i)=0;
           end
           
        end

    end
    
end


selectedFeatures=find(mode==1);

N=int16(sqrt(p));
if isempty(selectedFeatures)
    disp('sorted features');
    [~,I]=sort(depArray,'descend');
   selectedFeatures =I(1:N);
end

time=toc(start);    
end

function [ dep ] = dep_n1(data,Y,N)

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card(d,Y,class,N);
end
dep=card_ND/card_U;
end


function [c]=card(sets,Y,label,N)
         [~,I]=sort(sets);
         cNum=0;
         for j=2:N+1
             ind=I(j,1);
             if Y(ind,1)==label
                 cNum=cNum+1;
             end
         end  
         if label==0
             if cNum==N
                 c=1;
             else
                 c=0;
             end
         else
             c=cNum/N;
         end
end
