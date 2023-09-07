function [ selectedFeatures,time ] = SFSF_NR( X, Y,R)
% Classical Rough Set based framework 
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
     dep_single=dep_R(col,Y,R);                          
     depArray(1,i)=dep_single;

    if dep_single>=dep_Mean                                 
            mode(1,i)=1;             
            cols=X(:,mode==1);
            dep_New=dep_R(cols,Y,R);
            if dep_New>dep_Set                               
                dep_Set=dep_New;
                dep_Mean=sum(depArray(1,mode==1))/sum(mode);
                
           elseif dep_New==dep_Set                                                    
                 [index_del] = non_signf(X,Y,mode,R);
                 mode(1,index_del)=0;
                 dep_Mean=sum(depArray(1,mode==1))/sum(mode);              
            else
                mode(1,i)=0;    
            end
    
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);    
end

function [index_del] = non_signf(X,Y,mode,r)
%在备选集合中查找并移除significant==0的特征
B=zeros(1,length(mode));
R=mode;
indexs=find(mode==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind,r)==0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=find(B==1);
end

function [s]= sig(X,Y,mode,f_i,R)
%计算属性重要度
[d_B]=dep_R(X(:,mode==1),Y,R);
mode(1,f_i)=0;
[d_F]=dep_R(X(:,mode==1),Y,R);

s=(d_B-d_F)/d_B;
end


function [ dep ] = dep_R(data,Y,r)
%计算条件属性data对标签Y的依赖度

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card_R(d,Y,class,r);
end
dep=card_ND/card_U;
end

%根据最近邻N个备选的特征index，确定index对应记录的label是否相同
%根据相同程度计算card值
function [c]=card_R(sets,Y,label,R)
        max_d=max(sets,1);
        radius=R*max_d;
        
        set_r=find(sets<=radius);
        cTotal=length(set_r);
        cNum= length(find(Y(set_r,1)==label));
         
         if cTotal>0
            c=cNum/cTotal;
         else
             c=0;
         end
end