function [ selectedFeatures,time ] = OSFS_DD_z2(X,Y,sig1,sig2,PoolSize)
% 
% online streaming feature selection based on 3-way decision
% use fisher'z-test for continue   data
%
% Output:  selectedFeatures  ѡ���������ż���
%          time              �㷨����ʱ��
% Input:     X     �����������ݾ���
%            Y     ������ǩ����
%   

if nargin<3
    sig1=1;
    sig2=2;
    PoolSize=100;
end

if nargin<5
    PoolSize=100;
end

start=tic;
[~,P]=size(X);
AccIndexs=[];
DeterIndexs=[];

meanF=0;
stdF=0;
ZArray=[];

% gamma=0.01;

for i=1:P
    [~, fz_F] = Fisher_Ztest(X(:,i),Y);
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
        Acc_delIndx=checkPoolRedundancy(X,Y,AccIndexs,i,ZArray,beta);     
        if isempty(Acc_delIndx)
            continue;
        else
            delCount=length(Acc_delIndx);
            for d=1:delCount
                delInd=Acc_delIndx(1,d);
                ind= AccIndexs==delInd;
                AccIndexs(ind)=[];
            end
        end
    
    elseif(fz_F<=alpha)   
        %Reject��
        continue;
    else  
        %Determented��    
        DeterIndexs=[DeterIndexs,i]; 
        deterCount=length(DeterIndexs);
        if deterCount>=PoolSize
           newSelectedIndex=flushDeterPool(X,Y,DeterIndexs,beta,ZArray);
           AccIndexs=[AccIndexs,newSelectedIndex];
           DeterIndexs=[];
        end
    end

end

if ~isempty(DeterIndexs)
    newSelectedIndex=flushDeterPool(X,Y,DeterIndexs,beta,ZArray);
    AccIndexs=[AccIndexs,newSelectedIndex];
    DeterIndexs=[];
end

selectedFeatures=AccIndexs;
% disp('final selected feature indexs');
% disp(selectedFeatures);
% disp(length(selectedFeatures));
time=toc(start);   

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
        [CI_cmi,cmi]=Fisher_Ztest(X(:,ind),Y,X(:,ind_joint));
        if CI_cmi==1||isnan(cmi)
            continue;
        else
            jointMI=MIArray(1,ind)+cmi;        
            if jointMI>2*beta
                selectedIndex=[selectedIndex,ind,ind_joint];
                break;
            end
        end
    end
end
selectedIndex=unique(selectedIndex);
% disp('joint indexs');
% disp(selectedIndex);
end

%���Pool���Ƿ��������
function [delIndx]=checkPoolRedundancy(X,Y,AccIndexs,newIndex,MIArray,beta)

    delIndx=[];
    count=length(AccIndexs);
    if count>1    
        newMI=MIArray(1,newIndex);
        for i=1:(count-1)
            ind=AccIndexs(1,i);
            indMI=MIArray(1,ind);

            [CI_cmi,cmi]=Fisher_Ztest(X(:,ind),Y,X(:,newIndex));
            if CI_cmi==1||isnan(cmi)
                continue;
            else
                jointMI=cmi+indMI;
                 if jointMI<2*beta
                     if newMI<indMI
                         delIndx=[newIndex];
                         break;
                     else
                         delIndx=[delIndx,ind];
                     end
                 end
            end

        end

    end
end


