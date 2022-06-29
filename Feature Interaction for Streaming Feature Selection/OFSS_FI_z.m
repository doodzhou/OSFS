function [ selectedFeatures,time ] = OFSS_FI_z(X,Y,alpha,gamma)
%OFS_INTERACTION 
% online streaming feature selection considering feature interaction
% use Fisher's Z-test for continue data
%
% Output:  selectedFeatures  选择的特征序号集合
%                time   算法运行时间
% Input:  X       样本属性数据矩阵
%         Y       样本标签矩阵
%         alpha   Fisher Z-test重要度
%         gamma   交互度

start=tic;
[N,P]=size(X);
Mode=zeros(1,P);
Dep_Array=zeros(1,P);

Int_Array=zeros(P,P);

for i=1:P
    
    [CI, dep_F] = Fisher_Ztest(X(:,i),Y,[],alpha);
    if CI==1||isnan(dep_F)
        continue;
    else
        if sum(Mode)==0
            Dep_Array(1,i)=dep_F;
            Mode(1,i)=1;
            continue;
        end
    end
        
    [int_val,Int_Array]=mi_interaction(X,Y,Mode,i,alpha,Int_Array,Dep_Array);
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
%                     [CI_ij,mi_ij]=my_cond_indep_fisher_z(data,i, j, [],N,alpha);
                    [CI_ij,mi_ij]=Fisher_Ztest(X(:,i),X(:,j),[],alpha);
                    if CI_ij==1||isnan(mi_ij)
                        continue;
                    end                        
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


function [val,Int_Array]=mi_interaction(X,Y,Mode,newInd,alpha,Int_Array,dep_Array)
%计算i同已选特征的平均交互值

val=0;
selectedF=find(Mode==1);
len=length(selectedF);
sum=len;

if len>0
    for i=1:len
        ind=selectedF(:,i);
%         [CI_cmi,cmi]=my_cond_indep_fisher_z(data,ind, P, newInd,N,alpha);
        [CI_cmi,cmi]=Fisher_Ztest(X(:,ind),Y,X(:,newInd),alpha);
        mi=dep_Array(1,ind);
        
        if CI_cmi==1||isnan(cmi)||isnan(mi)   
            sum=sum-1;
           continue;
        else
            val=val+cmi-mi;
            Int_Array(ind,newInd)=cmi-mi;
        end
    end
    val=val/sum;
end

end

