function [ selectedFeatures,time ] = OFS_A3M( X, Y)
% Adapated Neighgborhood rough set 
% Max-dependency Max-relevance Max-significance
%
% Output:  selectedFeatures  选择的特征序号集合
%                time   算法运行时间
% Input:  X     样本属性数据矩阵
%            Y     样本标签矩阵
%            N     判别点的左右邻近点数
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
     dep_single=dep_an(col,Y);                          %根据样本的正域个数计算单个属性的依赖性
     depArray(1,i)=dep_single;
     
    if dep_single>dep_Mean                               % Max-relevance 单个特征依赖度>dep_mean 则成为备选特征
            mode(1,i)=1;             
            cols=X(:,mode==1);                               %计算联合贡献度    
            dep_New=dep_an(cols,Y);
            if dep_New>dep_Set                             %Map-dependency
                   dep_Set=dep_New;
                   dep_Mean=dep_Set/sum(mode);
            elseif dep_New==dep_Set                                                         %Max-significance

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
%在备选集合中查找并移除significant==0的特征
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
%计算属性重要度
[d_B]=dep_an(X(:,mode==1),Y);
mode(1,f_i)=0;
[d_F]=dep_an(X(:,mode==1),Y);

s=(d_B-d_F)/d_B;
end


function [ dep ] = dep_an(data,Y)
%计算条件属性data对标签Y的依赖度

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


%不需要参数N计算card值
%N个数为最近邻点数：使用最大距离-最小距离，然后分成N等份，如果由近及远的某一等份中无点，取前x等份中点数
%               [max{d}-min{d}]/N+x*min{d}    
%根据相同程度计算card值
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

