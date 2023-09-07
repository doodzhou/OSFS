function [ selectedFeatures,time ] = OSFS_DD_z2(X,Y,sig1,sig2,PoolSize)
% 
% online streaming feature selection based on 3-way decision
% use fisher'z-test for continue   data
%
% Output:  selectedFeatures  选择的特征序号集合
%          time              算法运行时间
% Input:     X     样本属性数据矩阵
%            Y     样本标签矩阵
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
    
    %前两个特征默认都选择
    if i<=2
        AccIndexs=[AccIndexs,i];
        meanF=mean(ZArray);
        stdF=std(ZArray);
        continue;
    end
    
    %更新mean和std
    [mean_new,std_new]=updateMeanStd(fz_F,meanF,stdF,i);
    meanF=mean_new;
    stdF=std_new;
    
    %更新alpha和beta值
    alpha=meanF+sig1*stdF;
    beta=meanF+sig2*stdF;  
    
    %Accept域
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
        %Reject域
        continue;
    else  
        %Determented域    
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

%更新均值和方差
function [mean_new,std_new]=updateMeanStd(newVal,mean,std,Num)
     mean_new=mean+(newVal-mean)/Num;
    if Num>2
        Sn=(1/(Num-1))*((Num-2)*std^2+((Num-1)/Num)*(newVal-mean)^2);
        std_new=sqrt(Sn);
    else
        std_new=0;
    end
end


%处理缓存池中的边界特征
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

%检测Pool中是否存在冗余
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


