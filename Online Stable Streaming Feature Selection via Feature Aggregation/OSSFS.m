function [ finalSelected,time ] = OSSFS( X,Y,alpha)
% Online stable streaming feature selection
% 
X=mapminmax(X);
start=tic;
[~,P]=size(X);
mode=zeros(1,P);
groupUpdateNum=8;
maxHyNum=8;
closestK = 7;

HyperE=struct('center',[],'R',[],'S',[],'radius',[]);

newAddingNum=0;
newAddingIndex=zeros(1,groupUpdateNum);
for i=1:P
    cols=X(:,i);
    %过滤无关特征
    [CI, dep_F] = Fisher_Ztest(cols,Y,[],alpha);
    if CI==1||isnan(dep_F)||dep_F<0.1
        continue;
    end
    
    mode(1,i)=1;
    newAddingNum=newAddingNum+1;
    newAddingIndex(1,newAddingNum)=i;
    %备选特征数每新增groupUpdateNum个更新一次
    if newAddingNum==groupUpdateNum
        if sum(mode)==groupUpdateNum
            %bandwidth估计
            [bandwidth]=estimate_bandwidth(X(:,mode==1)');

            HyperE=HyperECover(X(:,mode==1)', bandwidth);
            disp(bandwidth);
        else
            %根据新的点col更新Hcover, Hcover_id,sph_R,sph_S
            HyperE=HyperEUpdate(HyperE,X(:,newAddingIndex)',X(:,mode==1)',bandwidth);
        end
        newAddingNum=0;
        newAddingIndex=zeros(1,groupUpdateNum);
    end

end

%备选特征数过少
if sum(mode)<groupUpdateNum
    [bandwidth]=estimate_bandwidth(X(:,mode==1)');
    HyperE=HyperECover(X(:,mode==1)', bandwidth);
    disp(bandwidth);
end

% %最后剩余特征
if newAddingNum>0
    newAddingIndex=nonzeros(newAddingIndex)';
    HyperE=HyperEUpdate(HyperE,X(:,newAddingIndex)',X(:,mode==1)',bandwidth);
end

%合并超球体
[h_labels,NewMergeHyperE]=HyperEMerge(HyperE,maxHyNum,bandwidth);

finalSelected=[];
selectedFeatures=find(mode==1);
%从group中选择离中心最近的特征
clusterCenters=NewMergeHyperE.center;
clusterArray=unique(h_labels);
clusterNum=length(clusterArray);
disp('cluster Num');
disp(clusterNum);
for cl=1:clusterNum
    clusterID=clusterArray(cl);
    clusetInds=selectedFeatures(h_labels==clusterID);

    clusetFeatures=X(:,clusetInds)';
    clCenter=clusterCenters(clusterID,:);

    disArray=pdist2(clCenter,clusetFeatures);
    [~,sortInd]=sort(disArray);
    if length(sortInd)>closestK
        minInd=sortInd(1:closestK);
    else
        minInd=sortInd;
    end
    groupSelected=clusetInds(minInd);

    finalSelected=[finalSelected,groupSelected];
end

disp(finalSelected);

time=toc(start);
end


function [bandwidth]=estimate_bandwidth(Dataset)
%估计样本的bandwidth
[n,~]=size(Dataset);
bandwidth=0;

D = pdist(Dataset); %成对观测值之间的两两距离
DArray=squareform(D,'tomatrix');

for i=1:n
    d = DArray(:,i);
    sortD = sort(d);
    sortDIndex = round(length(sortD)/2);
    bandwidth=bandwidth+sortD(sortDIndex+1);
end

bandwidth=bandwidth/n;

end

function [h_label,NewHyperE]=HyperEMerge(HyperE,maxHyNum,bandwidth)
%合并超椭球

DataSet=HyperE.center;
h_label=HyperE.label;
insNum=length(h_label);
while 1
      [NewHyperE]=HyperECover(DataSet, bandwidth);
      it_label=NewHyperE.label;
      for j=1:insNum
            centerInd=h_label(j,1);
            h_label(j,1)=it_label(centerInd,1);
      end
      newCenters=NewHyperE.center;
%       disp(size(newCenters,1));
      if size(DataSet,1)==size(newCenters,1)||size(newCenters,1)<maxHyNum
          break;
      else
          DataSet =newCenters;
      end
end


end



function [NewHyperE]=HyperEUpdate(HyperE,newInstances,DataSet,bandwidth)
%超椭球体集合动态更新

sph_pStart=HyperE.pstart;
sph_center=HyperE.center;
sph_radius=HyperE.radius;
sph_R=HyperE.R;
sph_S=HyperE.S;

newAddNum=size(newInstances,1);
%更新R,S,center，radius
numH=size(sph_center,1);
for na=1:newAddNum
    newF=newInstances(na,:);
    for h=1:numH
        h_pStart=sph_pStart(h,:);
        weight_F = exp(-sum((h_pStart-newF).^2,2)/bandwidth^2);
        new_R=weight_F*newF+sph_R(h,:);
        new_S=weight_F+sph_S(h,:);
        new_center=new_R/new_S;
        new_radius=sqrt(sum((new_center-h_pStart).^2,2));

        sph_R(h,:)=new_R;
        sph_S(h,:)=new_S;
        sph_radius(h,:)=new_radius;
        sph_center(h,:)=new_center;
    end
end

%更新样本对应关系
numFSet=size(DataSet,1);
sph_label=zeros(numFSet,1);
for i=1:numFSet
       data_cur = DataSet(i,:);
       cover_flag = 0;
       for j=1:numH
                sph_cur_center = sph_center(j,:);
                sph_cur_radius = sph_radius(j,:);
                dist = sqrt(sum((data_cur-sph_cur_center).^2,2));
                 if (dist<sph_cur_radius)||(dist<1e-3)
                      sph_label(i,1) = j; 
                      cover_flag = 1;
                      break;
                 end
                
       end 

       %创建新的椭球体
       if cover_flag==0 
           %TODO  更新
            [s_center, s_radius,s_R,s_S] = HyperEConstruct(data_cur, DataSet, bandwidth); 
            sph_center = [sph_center; s_center];
            sph_radius=[sph_radius;s_radius];
            sph_R=[sph_R;s_R];
            sph_S=[sph_S;s_S];
            sph_pStart=[sph_pStart;data_cur];
            sph_label(i,1) = size(sph_center,1);
       end
end

NewHyperE=struct('pstart',sph_pStart,'center',sph_center,'R',sph_R,'S',sph_S,'radius',sph_radius,'label',sph_label);

end


function [HyperE]=HyperECover(DataSet, bandwidth)
%生成超椭球体集合
% ///////////////////////////////////////////////////////////////////////////////
% //   Input:
% //         Data_Query     - Query data matrix. Each column vector is a data point.
% //         Data_Ref       - Reference data matrix. Each column vector is a data point.
% //         sigma          - Bandwidth of Gaussian kernel
% //
% //   Output:
% //         sph_cov        - Constructed hyperspheres that cover the query data set.
% //         sph_cov_id     - Vector of covering hypersphere index. Each data point
% //							receives one index.   
% ///////////////////////////////////////////////////////////////////////////////


[num,dim] = size(DataSet);
sph_cov = [];
sph_R=[];
sph_S=[];
sph_pStart=[];
sph_cov_count = 0;
sph_cov_id = zeros(num,1);



% ------------- Scan the data set and sequentially cover it with hyperspheres ----------------
  for i=1:num
       data_cur = DataSet(i,:);
       cover_flag = 0;
       
       %// check whether the current point is covered by any exisitng hyperspheres
       for j=1:size(sph_cov,1) 
            sph_cur_center = sph_cov(j,1:dim);
            sph_cur_radius = sph_cov(j,dim+1);
            dist = sqrt(sum((data_cur-sph_cur_center).^2,2));
             if ((dist<sph_cur_radius)||(dist<1e-3))
                  sph_cov_id(i) = j; 
                  cover_flag = 1;
                  break;
             end
       end 
       
       %// if the current has not been covered yet, construct a new hypersphere
       if cover_flag==0
            [s_center, s_radius,s_R,s_S] = HyperEConstruct(data_cur, DataSet, bandwidth); 
            sph_cov = [sph_cov; [s_center, s_radius]];
            sph_R=[sph_R;s_R];
            sph_S=[sph_S;s_S];
            sph_pStart=[sph_pStart;data_cur];
            sph_cov_count = sph_cov_count+1;
            sph_cov_id(i) = sph_cov_count;
       end
              
  end
  
HyperE=struct('pstart',sph_pStart,'center',sph_cov(:,1:dim),'R',sph_R,'S',sph_S,'radius',sph_cov(:,dim+1),'label',sph_cov_id);

end

function [s_center, radius,h_R,h_S]=HyperEConstruct(p_start, data, sigma)
%超椭球构造
% ///////////////////////////////////////////////////////////////////////////////
% //   Input:
% //         p_start        - Given point from which hypersphere to be constructed
% //         data           - Data matrix. Each column vector is a data point.
% //         sigma          - Bandwidth of Gaussian kernel
% //
% //   Output:
% //         s_center       - Center of the constructed hypersphere.
% //         radius         - Radius of the cosntructed hypersphere.
% //         h_R              -vector accumulator
%//          h_S              -scalar accumulator       
% ///////////////////////////////////////////////////////////////////////////////
%此函数用于通过在数据集上运行一次 Mean-Shift 迭代来从给定点构造超球面。
%p_start 构建超球面的给定点
%s_center 所构建超球面的中心
%radius 所构建超球面半径
  p_start_rep = ones(size(data,1),1)*p_start;
 
  weight = exp(-sum((p_start_rep-data).^2,2)/sigma^2);
  weight_rep = weight*ones(1, size(data,2));
  h_R = sum(weight_rep.*data,1);
  h_S = sum(weight,1);

 s_center = h_R/h_S;
 radius = sqrt(sum((s_center-p_start).^2,2));
 
end

