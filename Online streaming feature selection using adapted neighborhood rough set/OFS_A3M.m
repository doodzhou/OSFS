function [ selectedFeatures,time ] = OFS_A3M( X, Y)
%  Peng Zhou;Xuegang Hu;Peipei Li 
% A New Online Feature Selection Method Using Neighborhood Rough Set
% 2017 IEEE International Conference on Big Knowledge (ICBK) 2017
%
% Output:  selectedFeatures  the index of selected features
%          time   running time
% Input:  X     data samples vector
%         Y     label vector
%   


start=tic;
[~,p]=size(X);

mode=zeros(1,p);                                             
dep_Mean=0;                                                    
dep_Set=0;                                                 
depArray=zeros(1,p);                                        


for i=1:p
     col=X(:,i);
     dep_single=dep_an(col,Y);                          
     depArray(1,i)=dep_single;
     
    if dep_single>dep_Mean                               
            mode(1,i)=1;             
            cols=X(:,mode==1);                                  
            dep_New=dep_an(cols,Y);
            if dep_New>dep_Set                             
                   dep_Set=dep_New;
                   dep_Mean=dep_Set/sum(mode);
            elseif dep_New==dep_Set                                                        
                   [index_del] = non_signf(X,Y,mode,i);
                   mode(1,index_del)=0;

                   dep_Mean=dep_Set/sum(mode);
           else
                    mode(1,i)=0;   
           end
    
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);    
end

function [index_del] = non_signf(X,Y,mode,i)

B=zeros(1,length(mode));
R=mode;
T=mode;
T(1,i)=0;

indexs=find(T==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind)==0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=B==1;
end

function [s]= sig(X,Y,mode,f_i)

[d_B]=dep_an(X(:,mode==1),Y);
mode(1,f_i)=0;
[d_F]=dep_an(X(:,mode==1),Y);

s=(d_B-d_F)/d_B;
end


function [ dep ] = dep_an(data,Y)

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card(d,Y,class,n);
end
dep=card_ND/card_U;
end


function [c]=card(sets,Y,label,N)
        [D,I]=sort(sets);        
        
        min_d=D(2,1);
        max_d=D(N,1);
        mean_d=1.5*(max_d-min_d)/(N-2);
        
        cNum=0;
        cTotal=1;
        ind2=I(2,1);
        if Y(ind2,1)==label
               cNum=cNum+1;
        end
        
         for j=3:N
             if D(j,1)-D(j-1,1)<mean_d
                 ind=I(j,1);
                 cTotal=cTotal+1;
                 if Y(ind,1)==label
                     cNum=cNum+1;
                 end
             else
                  break;
             end
         end  
        
         c=cNum/cTotal;

end

