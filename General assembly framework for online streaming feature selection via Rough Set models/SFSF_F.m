function [ selectedFeatures,time ] = SFSF_F( X,Y)
%FRS_OSFS_MIX 
% 基于FUZZY ROUGH SET理论构建的在线流特征选择算法
%INPUT      X：condition feature set
%           Y: decision feature set
%              

start=tic;

[~,p]=size(X);
mode=zeros(1,p); 

depMean=0;                                                
depSet=0;                                                  
Dep_Array=zeros(1,p);                                      
Sim_matrix=cell(1,p);                                       

for i=1:p
     Sim_matrix{1,i}=simFunSingle(X(:,i));
     dep_f_t=depDegree(Sim_matrix,i,Y);                    
     Dep_Array(1,i)=dep_f_t;     
     
    if dep_f_t>depMean                      
        mode(1,i)=1;       
        dep_New=round(depDegree(Sim_matrix,find(mode==1),Y),4);           
            
        if dep_New>depSet                            
               depSet=dep_New;
               depMean=sum(Dep_Array(1,mode==1))/sum(mode);
               
        elseif dep_New==depSet
               [index_del] = nonSig(Sim_matrix,Y,mode);
               mode(1,index_del)=0;
               depMean=sum(Dep_Array(1,mode==1))/sum(mode); 
               
        else
             mode(1,i)=0;
        end
        
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);
end

function [index_del] = nonSig(S,Y,mode)
%在备选集合中查找并移除significant==0的特征
%
B=zeros(1,length(mode));
R=mode;

indexs=find(mode==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(S,Y,R,ind)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=find(B==1);
end

function [s]= sig(S,Y,mode,f_i)
%计算属性重要度
%

[d_B]=depDegree(S,find(mode==1),Y);
mode(1,f_i)=0;
[d_F]=depDegree(S,find(mode==1),Y);

s=(d_B-d_F);

end


function[dependency]=depDegree(S,index,Y)
%基于模糊粗糙集的集合依赖度计算
%

p=length(index);
S_Array=S{1,index(1)};

if p>1 
   for i=2:p    
     S_Array=min(S_Array,S{1,index(i)});
   end
end


%uni_Y=unique(Y);
% num_Y=length(uni_Y);
% pos_Array=zeros(num_Y,n);
% I=ones(n,n);
% for j=1:num_Y
%     class=uni_Y(j);
%     s_Y=zeros(n,1);
%     s_Y(Y==class,1)=1;
%     F=I-S_Array+repmat( s_Y , 1 , n );
%     F(F>1)=1;
%     pos_Array(j,:)=min(F);
% end 
% pos=max(pos_Array);   
% dependency=sum(pos)/n;

[N,~]=size(S_Array);
lamuda=[];
 for i=1:N
    b1= Y~=Y(i);
    lowapp=min(1-S_Array(i,b1));%%%%FRS下近似公式
    lamuda=[lamuda,lowapp];%%%%%属性下的相似矩阵的，各个样本的下近似，得到一个行向量
end
 dependency=sum(lamuda)/N;

end

function [S_Array] = simFunSingle(F)
% 计算单个特征的模糊相似矩阵
% real-valued: u(x,y)=1- |a(x)-a(y)|/[a(max)-a(min)]
% nominal: 1  a(x)==a(y); 0 otherwise 

a_max=max(F);
a_min=min(F);
interval=abs(a_max-a_min);
[n,~]=size(F);

S_Array=zeros(n,n);
for i=1:n
    for j=i:n
        if(i==j)
            S_Array(i,j)=1;
        else
            if interval>0
                val=1-abs(F(j)-F(i))/interval;
            else
                val=1;
            end
            S_Array(i,j)=val;
            S_Array(j,i)=val;
        end       
    end
end

end


function pos=frs_pos(sim_matrix,d)
m=length(d);
lamuda=[];
   for i=1:m
    b1=find(d~=d(i));
    lowapp=min(1-sim_matrix(i,b1));%%%%FRS下近似公式
     lamuda=[lamuda,lowapp];%%%%%属性下的相似矩阵的，各个样本的下近似，得到一个行向量
   end
 pos=sum(lamuda)/m;
  
end


function [S_Array] = simFunSingle2(F,type)
% fuzzy similarity relations 
% std(a)=sqrt(var(a));
% u(x,y)=max{min{[a(y)-a(x)+std(a)]/std(a),[a(x)+std(a)-a(y)]/std(a)},0}

std=sqrt(var(F));
[n,~]=size(F);

S_Array=zeros(n,n);
for i=1:n
    for j=i:n
        if(i==j)
            S_Array(i,j)=1;
        else
            if type==0 
                val=0;
                if F(j)==F(i)
                    val=1;
                end
                S_Array(i,j)=val;
                S_Array(j,i)=val;
            else           
                min_1=(F(j)-F(i)+std)/std;
                min_2=(F(i)+std-F(j))/std;
                min=min_1;
                if min_2<min_1
                    min=min_2;
                end
                if min<0
                    min=0;
                end            
                S_Array(i,j)=min;
                S_Array(j,i)=min;
            end
        end
       
    end
end
end
