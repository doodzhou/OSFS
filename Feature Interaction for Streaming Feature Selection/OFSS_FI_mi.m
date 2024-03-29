function [ selectedFeatures,time ] = OFSS_FI_mi(X,Y,gamma)
%OFS_INTERACTION 
% online streaming feature selection considering feature interaction
% use mutual information for discrete   data
%
% Output:  selectedFeatures  选择的特征序号集合
%                time   算法运行时间
% Input:  X     样本属性数据矩阵
%         Y     样本标签矩阵
%         gamma  交互度
%           

start=tic;
[~,P]=size(X);
Mode=zeros(1,P);
Dep_Array=zeros(1,P);
data=[X,Y];

Int_Array=zeros(P,P);

for i=1:P
    
    dep_F = SU(X(:,i),Y);
    if dep_F==0
        continue;
    else
        if sum(Mode)==0
            Dep_Array(1,i)=dep_F;
            Mode(1,i)=1;
            continue;
        end
    end
        
    [int_val,Int_Array]=mi_interaction(data,Mode,i,Int_Array,Dep_Array);
%     disp([num2str(i),'...',num2str(int_val)]);
    %delete irrelevant features
    if int_val>=gamma
        Mode(1,i)=1;
        Dep_Array(1,i)=dep_F;

    elseif int_val>0
        Mode(1,i)=1;
        Dep_Array(1,i)=dep_F;
        selectedFeatures=find(Mode==1);
        p=length(selectedFeatures);
        %delete redundancy features  
        if p>1
            for j=1:p-1
                ind=selectedFeatures(1,j);
                if Int_Array(ind,j)>0
                    continue;
                else
                    mi_ij=SU(X(:,i),X(:,j));                       
                    if Dep_Array(ind)>dep_F && mi_ij>Dep_Array(i)
                           Mode(1,i)=0;
                           Dep_Array(1,i)=0;
                           break;
                    end   
                     if dep_F>Dep_Array(ind) && mi_ij>Dep_Array(ind)
                          Mode(1,ind)=0;
                          Dep_Array(1,ind)=0; 
                     end  
                end
            end
            
        end
        
    end

end

selectedFeatures=find(Mode==1);
time=toc(start);   

end


function [val,Int_Array]=mi_interaction(data,Mode,newInd,Int_Array,dep_Array)
%计算i同已选特征的平均交互值

[~,P]=size(data);
val=0;
selectedF=find(Mode==1);
len=length(selectedF);
sum=len;

if len>0
    for i=1:len
        ind=selectedF(:,i);
        cmi_v=cmi(data(:,ind), data(:,P), data(:,newInd));
        mi_v=dep_Array(1,ind);

        val=val+cmi_v-mi_v;
        Int_Array(ind,newInd)=cmi_v-mi_v;
    end
    val=val/sum;
end

end

