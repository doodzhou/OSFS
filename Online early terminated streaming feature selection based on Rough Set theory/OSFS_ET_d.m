function [selectedFeatures,time,terminateR] = OSFS_ET_d(X,Y,waitNum,delta,pauseTime)
% ���ھ���ֲڼ�������ѡ���㷨��Eearly Terminated�汾
% Input:   X   ѵ�����ݼ�
%          Y   ѵ�����ݼ���������
%  Output�� 
%
%
%

%��������ʱ��
[~,p]=size(X);

mode=zeros(1,p); 

dep_Set=0;                                                    
depSArray=[]; 
FLAG=0;
count=0;

start=tic;
for i=1:p
     if pauseTime>0
        pause(pauseTime);
     end
     col=X(:,i);
     dep_col=dep_D(col,Y);
     
     if dep_col>0
         mode(1,i)=1;             
         cols=X(:,mode==1);
         dep_New=dep_D(cols,Y);
     
        if dep_New>dep_Set       
            dep_Set=dep_New;
            depSArray=[depSArray,dep_New]; 
            if dep_Set==1
                terminateR=round(i/p,2);
                break;
            end
%             disp([i,dep_Set]);
        else  
           count=count+1;
%            [index_del] = non_signf(X,Y,mode,i);
%            disp("redundent features");
%            disp(find(index_del==1));
%            mode(1,index_del)=0;
        end
     end
     
    if count>=waitNum
        disp("==========ET Check=========")
         [FLAG]=SFS_ET(depSArray,waitNum,delta);
          if FLAG==1
            terminateR=round(i/p,2);
            break;
          end
     end
    
end
if FLAG==0
    terminateR=1;
end
selectedFeatures=find(mode==1);    
time=toc(start);

end

function [index_del] = non_signf(X,Y,mode,i)
%�ڱ�ѡ�����в��Ҳ��Ƴ�significant==0������
%
B=zeros(1,length(mode));
R=mode;
T=mode;
T(1,i)=0;

indexs=find(T==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    else
        R(1,ind)=1;
    end
end
index_del=B==1;
end

function [s]= sig(X,Y,mode,f_i)
%����������Ҫ��
%

[d_B]=dep_D(X(:,mode==1),Y);
mode(1,f_i)=0;
[d_F]=dep_D(X(:,mode==1),Y);

s=(d_B-d_F);
end


function [ dep ] = dep_D( X,Y )
%��ɢ�����ݵ���������ֵ����
%   Detailed explanation goes here

N=size(Y,1);
[POS,~,~]=RS_appr(X,Y);
%TODO ����4λ��Ч����
dep=round(length(POS)/N,4);
% dep=length(POS)/N;
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


