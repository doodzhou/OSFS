function [ selectedFeatures,time,DeterIndexs] = DHSFS_subnode(X,Y,sig1,sig2)
% 
% Online Distributed Heterogeneous Streaming Feature Selection
% 
%
% Output:  selectedFeatures  ѡ���������ż���
%          time              �㷨����ʱ��
% Input:     X     ����������
%            Y     ��ǩ����
%   

if nargin<3
    sig1=1;
    sig2=2;
 
end

start=tic;
[~,P]=size(X);
AccIndexs=[];
DeterIndexs=[];

meanF=0;
stdF=0;
ZArray=[];

for i=1:P
    [fz_F] = MIC_an(X(:,i),Y);
    if isnan(fz_F)
        fz_F=0;
    end
    ZArray=[ZArray,fz_F];
    
    %ǰ��������Ĭ�϶�ѡ��
    if i<=2
        AccIndexs=[AccIndexs,i];
        meanF=mean(ZArray);
        stdF=std(ZArray);
        continue;
    end
    
    %����mean��std
    [mean_new,std_new]=updateMeanStd(fz_F,meanF,stdF,i);
    meanF=mean_new;
    stdF=std_new;
    
    %����alpha��betaֵ
    alpha=meanF+sig1*stdF;
    beta=meanF+sig2*stdF;  
    
    %Accept��
    if(fz_F>=beta)
        AccIndexs=[AccIndexs,i];
    
    elseif(fz_F<=alpha)   
        %Reject��
        continue;
    else  
        %Determented��    
        DeterIndexs=[DeterIndexs,i]; 
       continue;
    end

end

selectedFeatures=AccIndexs;

time=toc(start);   

end



function[dep_mic]=MIC_an(data,Y)
data1=data';
Y1=Y';
dep = mine(data1,Y1);
dep_mic=dep.mic;

end

%���¾�ֵ�ͷ���
function [mean_new,std_new]=updateMeanStd(newVal,mean,std,Num)
     mean_new=mean+(newVal-mean)/Num;
    if Num>2
        Sn=(1/(Num-1))*((Num-2)*std^2+((Num-1)/Num)*(newVal-mean)^2);
        std_new=sqrt(Sn);
    else
        std_new=0;
    end
end


%��������еı߽�����
function [selectedIndex]=flushDeterPool(X,Y,DeterIndexs,beta,MIArray)

DeterCount=length(DeterIndexs);
selectedIndex=[];

for i=1:DeterCount
    ind=DeterIndexs(1,i);
    for j=(i+1):DeterCount
        ind_joint=DeterIndexs(1,j);
        jointMI=MIArray(1,ind)+MIC_an(X(:,ind),X(:,ind_joint));
        if jointMI>2*beta
            selectedIndex=[selectedIndex,ind,ind_joint];
            break;
        end
    end
end

selectedIndex=unique(selectedIndex);
end

%���Pool���Ƿ��������
function [delIndx]=checkPoolRedundancy(X,Y,AccIndexs,newIndex,MIArray,beta)
    delIndx=[];
    count=length(AccIndexs);
    if count>1    
%         newMI��ʾ�µ���pool�ص������Ļ���Ϣ��С
        newMI=MIArray(1,newIndex); 
        for i=1:(count-1)
            ind=AccIndexs(1,i);
            indMI=MIArray(1,ind);
            
            jointMI=MIC_an(X(:,ind),X(:,newIndex));
            if(jointMI>indMI && jointMI>newMI)
                if(newMI < indMI)
                    delIndx=[newIndex];
                    break;
                else
                    delIndx=[delIndx,ind];
                end
            else
                delIndx=delIndx;
            end
            
       end

    end
end


