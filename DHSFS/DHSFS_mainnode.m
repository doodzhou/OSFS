function [se_F,Ttime] = DHSFS_mainnode(X,Y,sig1,sig2)
% Online Distributed Heterogeneous Streaming Feature Selection
% Output:  se_F  选择的特征序号集合
%          Ttime              算法运行时间
% Input:     X     特征流矩阵
%            Y     标签矩阵
%     

if nargin<3
    sig1=1;
    sig2=2;
end

[~,data_c]=size(X);
indices = randperm(data_c);

MIArray_deF=[];
MIArray_seF=[];
Mean_seF=0;
Std_seF=0;

%   单个分节点的结果  
[selectedFeatures,time,DeterIndexs] = DHSFS_subnode(X(:,k),Y,sig1,sig2);
node_sf=cell2mat(selectedFeatures);
node_df=cell2mat(DeterIndexs);
sele_F{k}=node_index(1,node_sf);
dete_F{k}=node_index(1,node_df);
se_F=cell2mat(sele_F);
de_F=cell2mat(dete_F);
time_s=max(cell2mat(time));
 
%     计算总节点的均值和方差
for i=se_F
   mi_F = MIC_an(X(:,i),Y);
   MIArray_seF=[MIArray_seF,mi_F];
end
Mean_seF=mean(MIArray_seF);
Std_seF=std(MIArray_seF);


%     更新总节点的beta阈值
%     alpha=Mean_seF+sig1*Std_seF;
beta=Mean_seF+2*Std_seF;

%     对所有延迟区域的特征进行进一步筛选
start=tic;
for i=de_F
   mi_F = MIC_an(X(:,i),Y);
   if mi_F >= beta
        se_F=[se_F,i];
   else
       continue;
   end
end
time_d=toc(start); 
Ttime=time_s;
end

function[dep_mic]=MIC_an(data,Y)
% 计算两个变量之间的MIC的值
data1=data';
Y1=Y';
dep = mine(data1,Y1);
dep_mic=dep.mic;
end

