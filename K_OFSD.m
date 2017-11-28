function [ selectedFeatures,time ] = K_OFSD( X, Y,N)
% 使用K个邻居的类标签属性信息，运用粗糙集理论进行特征选择
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
dep_last=0;                                                       %目前已选特征的依赖度

depArray=zeros(1,p);                                       %单个依赖度数组
% W=length(find(Y==0))/length(Y);
% disp(W);

alpha=0.5;

for i=1:p
    col=X(:,i);
    dep_single=dep_n1(col,Y,N);                  %根据样本的正域个数计算单个属性的依赖性
    depArray(1,i)=dep_single;
    
    if dep_single>alpha                                      %单个特征依赖度>阀值 则成为备选特征
        
        if dep_single>dep_last
%             disp(dep_single);
%             disp(i);
            mode((mode==1))=0;
            mode(1,i)=1;
            dep_last=dep_single;                                 %计算整体贡献度  
        else 
           mode(1,i)=1;             
           cols=X(:,mode==1);                               %计算联合贡献度    
           dep_all=dep_n1(cols,Y,N);
           if dep_all>dep_last                                 %提高整体贡献度
                dep_last=dep_all;
%                 disp([dep_single,dep_last]);
           else
                 mode(1,i)=0;
           end
           
        end

    end
    
end

% disp(dep_last);
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
%计算条件属性data对标签Y的依赖度
%计算每个节点左右区域内N个节点的标签相同个数比weight
%或者如果正例个数大于负例个数，card+1

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

%根据最近邻N个备选的特征index，确定index对应记录的label是否相同
%根据相同程度计算card值
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
