function [ selectedFeatures,time ] = SFSF_C( X, Y)
% Classical Rough Set based framework 
%
%  
%   

%��������ʱ��
start=tic;
[~,p]=size(X);

mode=zeros(1,p);                                             %���ѡ�е������±�
dep_Mean=0;                                                    %Ŀǰ��ѡ�������������������dep��ֵ
dep_Set=0;                                                        %Ŀǰ��ѡ�������ϵ�depֵ
depArray=zeros(1,p);                                        %��������������


for i=1:p
     col=X(:,i);
     dep_single=dep_an(col,Y);                          
     depArray(1,i)=dep_single;

    if dep_single>dep_Mean*0.5                                 
            mode(1,i)=1;             
            cols=X(:,mode==1);
            dep_New=dep_an(cols,Y);
            if dep_New>dep_Set                              
                dep_Set=dep_New;
                dep_Mean=sum(depArray(1,mode==1))/sum(mode);
                
            elseif dep_New==dep_Set                                                        
                 [index_del] = non_signf(X,Y,mode);
                 mode(1,index_del)=0;
                 dep_Mean=sum(depArray(1,mode==1))/sum(mode);               
            else
                 mode(1,i)=0;   
            end
    
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);    
end

function [index_del] = non_signf(X,Y,mode)
%�ڱ�ѡ�����в��Ҳ��Ƴ�significant==0������
B=zeros(1,length(mode));
R=mode;

indexs=find(mode==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=find(B==1);
end

function [s]= sig(X,Y,mode,f_i)
%����������Ҫ��
[d_B]=dep_an(X(:,mode==1),Y);
mode(1,f_i)=0;
[d_F]=dep_an(X(:,mode==1),Y);

s=d_B-d_F;
end


function [ dep ] = dep_an(X,Y)
%������������data�Ա�ǩY��������
N=size(Y,1);
[POS,~,~]=RS_appr(X,Y);
dep=length(POS)/N;

end

function [POS,BND,UP]=RS_appr(B,Y)
%�����Ͻ��ơ��½��ƺͱ߽�
% Input: B ��������  X �����ռ伯��
% Output:POS �½��� UP �Ͻ��� BND �߽�
%
[un_B,~,ic_B]=unique(B,'rows');
[n,~]=size(B);

POS=zeros(n,1);
UP=zeros(n,1);
PB=size(un_B,1);

[un_Y,~,ic_Y]=unique(Y,'rows');
PY=size(un_Y,1);

for j=1:PY
    indx_Y=find(ic_Y==j);
    for i=1:PB
        indx=find(ic_B==i);
        lia = ismember(indx,indx_Y);
        if sum(lia)>0&&sum(lia)==length(lia)
            POS(indx,1)=1;
        end
        if sum(lia)>0
            UP(indx,1)=1;
        end
    end
end
 BND=find(POS~=UP);
 POS=find(POS==1);
 UP=find(UP==1);

end



