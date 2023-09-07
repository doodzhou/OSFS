function [ selectedFeatures,time ] = SFSF_NK( X, Y,K)
% k-nearest Neighborhood Rough Set based framework 
%
%  
%   

%计算运行时间
start=tic;
[~,p]=size(X);

mode=zeros(1,p);                                             %标记选中的属性下标
dep_Mean=0;                                                    %目前已选特征集合里面个特征的dep均值
dep_Set=0;                                                        %目前已选特征集合的dep值
depArray=zeros(1,p);                                        %单个依赖度数组


for i=1:p
     col=X(:,i);
     dep_single=dep_K(col,Y,K);                          
     depArray(1,i)=dep_single;

    if dep_single>=dep_Mean                                 
            mode(1,i)=1;             
            cols=X(:,mode==1);
            dep_New=dep_K(cols,Y,K);
            if dep_New>dep_Set                               
                dep_Set=dep_New;
                dep_Mean=sum(depArray(1,mode==1))/sum(mode);
                
            elseif dep_New==dep_Set                                                       
                 [index_del] = non_signf(X,Y,mode,K);
                 mode(1,index_del)=0;
%                  disp(index_del);
                 dep_Mean=sum(depArray(1,mode==1))/sum(mode);              
            else
                mode(1,i)=0; 
            end
    
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);    
end

function [index_del] = non_signf(X,Y,mode,K)
%在备选集合中查找并移除significant==0的特征
B=zeros(1,length(mode));
R=mode;
indexs=find(mode==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind,K)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=find(B==1);
end

function [s]= sig(X,Y,mode,f_i,K)
%计算属性重要度
[d_B]=dep_K(X(:,mode==1),Y,K);
mode(1,f_i)=0;
[d_F]=dep_K(X(:,mode==1),Y,K);

s=d_B-d_F;
end


function [ dep ] = dep_K(data,Y,K)
%计算条件属性data对标签Y的依赖度

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card_K(d,Y,class,K);
end
dep=card_ND/card_U;
end


function [c]=card_K(distArray,Y,label,K)
% K个最近邻居标签信息
%
        [~,I]=sort(distArray);        
        cNum=0;
        for i=1:K
            ind=I(i);
            if Y(ind)==label
               cNum=cNum+1;
            end
        end
        c=cNum/K;
end

