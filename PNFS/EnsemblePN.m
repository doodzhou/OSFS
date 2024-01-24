function [resultPN, resultSVM, resultKNN, resultTREE, resultRF, vote] = EnsemblePN(result_P, result_N, rowTest, claTotal, labelSize, PNRatio)
%ENSEMBLEPN 此处显示有关此函数的摘要
%此处显示详细说明
%claTotal分类器个数

%用正负特征
resultPN = [];
%只用正特征
%resultP = [];

%SVM
resultSVM = [];
%KNN
resultKNN = [];
%TREE
resultTREE = [];
%RF
%resultRF= [];
%C4.5
%resultC4_5= [];


for rowIndex = 1:rowTest
 
    %vote投票结果矩阵
    vote_P = [];  
    vote_N = [];

    %不同分类器的结果的合并成一个矩阵
    %第一行KNN的结果第二行SVM的结果...
    P_Matrix = [];
    N_Matrix = [];

    %处理Result_P
    for RPIndex = 1: claTotal
        %合并多个分类器的结果到P_Matrix
        %三个分类器
        P_Matrix = [result_P{rowIndex,1}; result_P{rowIndex,2}; result_P{rowIndex,3}];
        %四个分类器
        %P_Matrix = [result_P{rowIndex,1}; result_P{rowIndex,2}; result_P{rowIndex,3}; result_P{rowIndex,4}];
        %五个分类器
        %P_Matrix = [result_P{rowIndex,1}; result_P{rowIndex,2}; result_P{rowIndex,3}; result_P{rowIndex,4}; result_P{rowIndex,5}];
    end


    if PNRatio ~= 1
         %处理Result_N
        for RNIndex = 1: claTotal
            %三个分类器
            N_Matrix = [result_N{rowIndex,1}; result_N{rowIndex,2}; result_N{rowIndex,3}]; 
            %四个分类器
            %N_Matrix = [result_N{rowIndex,1}; result_N{rowIndex,2}; result_N{rowIndex,3}; result_N{rowIndex,4}]; 
            %五个分类器
            %N_Matrix = [result_N{rowIndex,1}; result_N{rowIndex,2}; result_N{rowIndex,3}; result_N{rowIndex,4}; result_N{rowIndex,5}]; 
        end
    end

    
    for i = 1: labelSize
        vote_P =  [vote_P, length(find(P_Matrix(:,i) == i))];
        if PNRatio ~= 1
            vote_N =  [vote_N, length(find(N_Matrix(:,i) == 0))];
        end 
    end

     
    if PNRatio ~= 1
         vote{rowIndex, 1} = vote_P + vote_N;
    else
         vote{rowIndex, 1} = vote_P;
    end
   

    %得出vote_P的结果
    %resultP(rowIndex, 1) = GetVote(vote_P);

     %得出vote的结果
     resultPN(rowIndex, 1) = GetVote(vote{rowIndex, 1});
    

    %得出其他分类器单独的结果
    %这里分类器的结果只用了正特征
    resultKNN(rowIndex, 1) = GetVote(result_P{rowIndex, 1});
    resultSVM(rowIndex, 1) = GetVote(result_P{rowIndex, 2});
    resultTREE(rowIndex, 1) = GetVote(result_P{rowIndex, 3});

end

end

