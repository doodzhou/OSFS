function [se_F,Ttime] = DHSFS_mainnode(X,Y,sig1,sig2)
% Online Distributed Heterogeneous Streaming Feature Selection
% Output:  se_F  ѡ���������ż���
%          Ttime              �㷨����ʱ��
% Input:     X     ����������
%            Y     ��ǩ����
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

%   �����ֽڵ�Ľ��  
[selectedFeatures,time,DeterIndexs] = DHSFS_subnode(X(:,k),Y,sig1,sig2);
node_sf=cell2mat(selectedFeatures);
node_df=cell2mat(DeterIndexs);
sele_F{k}=node_index(1,node_sf);
dete_F{k}=node_index(1,node_df);
se_F=cell2mat(sele_F);
de_F=cell2mat(dete_F);
time_s=max(cell2mat(time));
 
%     �����ܽڵ�ľ�ֵ�ͷ���
for i=se_F
   mi_F = MIC_an(X(:,i),Y);
   MIArray_seF=[MIArray_seF,mi_F];
end
Mean_seF=mean(MIArray_seF);
Std_seF=std(MIArray_seF);


%     �����ܽڵ��beta��ֵ
%     alpha=Mean_seF+sig1*Std_seF;
beta=Mean_seF+2*Std_seF;

%     �������ӳ�������������н�һ��ɸѡ
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
% ������������֮���MIC��ֵ
data1=data';
Y1=Y';
dep = mine(data1,Y1);
dep_mic=dep.mic;
end

