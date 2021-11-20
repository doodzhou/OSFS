function [ selectedFeatures,time,terminateR,dep_Set] = OSFS_ET_c( X,Y,K,width,increase,pauseTime)
% 基于K-nearest neighborhood relation的Early Terminated Online Streaming Feature Selection算法方法
% Input: 
%       X: train instances
%       Y: train labels
%       width: the number of features expected to wait;
%       increase: the dependency degree expected to increase;
%       pauseTime: simulate the streaming features generated at a certain time interval 
% Output:
%       selectedFeatures: the indexs of selected fetaures in X
%       time: runing time


 if nargin<5
    increase=0.01;
    pauseTime=0;
 end

 if nargin<6
    pauseTime=0;
 end

%计算运行时间
start=tic;
[~,p]=size(X);

mode=zeros(1,p);                                                                                          
dep_Set=0;   

depSArray=[];       
FLAG=0;
count=0;

for i=1:p      
     mode(1,i)=1;
     cols=X(:,mode==1);
     if pauseTime>0
        pause(pauseTime);
     end
     dep_New=dep_K(cols,Y,K);
     depSArray=[depSArray,dep_New];
     
     if dep_New>dep_Set 
        dep_Set=dep_New;
        count=0;
     else
         mode(1,i)=0;
         count=count+1;
         if mod(count,100)==0
             [index_del] = non_signf(X,Y,mode,i,K);
             mode(1,index_del)=0;
         end
     end
     
     if length(depSArray)>width
         startInd=i-width;
         if (dep_New-depSArray(1,startInd))<increase
            FLAG=1;
            terminateR=round(i/p,2);
            [index_del] = non_signf(X,Y,mode,i,K);
            mode(1,index_del)=0;

            time=toc(start);
            selectedFeatures=find(mode==1);
            break;
         end
     end
     
end

if FLAG==0
    terminateR=1;            
    [index_del] = non_signf(X,Y,mode,i,K);
    mode(1,index_del)=0;
    selectedFeatures=find(mode==1);
    time=toc(start);    
end

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

function [index_del] = non_signf(X,Y,mode,i,K)
%在备选集合中查找并移除significant==0的特征
%
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
    if sig(X,Y,R,ind,K)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    else
        R(1,ind)=1;
    end
end
index_del=B==1;
end

function [s]= sig(X,Y,mode,f_i,K)
%计算属性重要度
%

[d_B]=dep_K(X(:,mode==1),Y,K);
mode(1,f_i)=0;
[d_F]=dep_K(X(:,mode==1),Y,K);

s=(d_B-d_F);
end
