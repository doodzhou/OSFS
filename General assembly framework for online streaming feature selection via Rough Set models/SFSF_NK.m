function [ selectedFeatures,time ] = SFSF_NK( X, Y,K)
% k-nearest Neighborhood Rough Set based framework 
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
     dep_single=dep_K(col,Y,K);                          
     depArray(1,i)=dep_single;

    if dep_single>=dep_Mean                                 
            mode(1,i)=1;             
            cols=X(:,mode==1);
            dep_New=dep_K(cols,Y,K);
            if dep_New>dep_Set                               
                dep_Set=dep_New;
                dep_Mean=sum(depArray(1,mode==1))/sum(mode);
                
            elseif dep_New==dep_Set                                                       
                 [index_del] = non_signf(X,Y,mode,K);
                 mode(1,index_del)=0;
%                  disp(index_del);
                 dep_Mean=sum(depArray(1,mode==1))/sum(mode);              
            else
                mode(1,i)=0; 
            end
    
     end
    
end

selectedFeatures=find(mode==1);

time=toc(start);    
end

function [index_del] = non_signf(X,Y,mode,K)
%�ڱ�ѡ�����в��Ҳ��Ƴ�significant==0������
B=zeros(1,length(mode));
R=mode;
indexs=find(mode==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind,K)<=0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=find(B==1);
end

function [s]= sig(X,Y,mode,f_i,K)
%����������Ҫ��
[d_B]=dep_K(X(:,mode==1),Y,K);
mode(1,f_i)=0;
[d_F]=dep_K(X(:,mode==1),Y,K);

s=d_B-d_F;
end


function [ dep ] = dep_K(data,Y,K)
%������������data�Ա�ǩY��������

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card_K(d,Y,class,K);
end
dep=card_ND/card_U;
end


function [c]=card_K(distArray,Y,label,K)
% K������ھӱ�ǩ��Ϣ
%
        [~,I]=sort(distArray);        
        cNum=0;
        for i=1:K
            ind=I(i);
            if Y(ind)==label
               cNum=cNum+1;
            end
        end
        c=cNum/K;
end

