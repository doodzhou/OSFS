function [PNcoe, selectedFeaturesMatrix, weightMatrix, PNRatio, Mytime] = CalPN(DataSet, label, classNum, DUM)
% KENDALL 此处显示有关此函数的摘要
% 此处显示详细说明
% data 数据集
% label 标签向量
% WeightMatrix 是选择出来的特征的权重

%加1去除0标签
%label = label + 1;

start=tic;

PNcoe = {};
kendallMatrix = [];
selectedFeaturesMatrix = {};%第一行为正特征第二行为负特征
weightMatrix={};
class = {};

%获得数据集特征的个数和数据集的行数
[dataRow, featureIndex] = size(DataSet);

%过滤其他分类标签
for classIndex = 1 : classNum
    temp = label;
    temp(temp ~= classIndex) = 0;
    class{classIndex} = temp;
end


for classnum = 1 : classNum
    for i = 1 : featureIndex
        data = [DataSet(:, i), class{classnum}];
        [result,~]=corr(data, 'Type', 'Kendall');
        kendallMatrix(i, classnum) = result(1, 2);
    end
end


PNRatio = GetPNRatio(kendallMatrix);
%自动输出PNRatio


%需要选取的特征排名范围，前n个特征
%TO DO 变成自适应调整
%rankLength = 5;

%正负特征比例计算个数
rankPLength = floor(DUM * PNRatio);

if rankPLength < 2
    rankPLength = 5;
end

rankNLength = DUM - rankPLength;


%排序选取正负特征
for columIndex = 1:classNum

    %根据排序选取排名靠前的特征
    [colum, pos] =  sort(kendallMatrix(:, columIndex),'descend');

    %消除NaN
    matrix =  [colum, pos];
    nanRows = any(isnan(matrix), 2);
    matrixWithoutNaN = matrix(~nanRows, :);
    pos = matrixWithoutNaN(:,2)';
    [~, posLenth] = size(pos);
    [rowNum, ~] = size(matrixWithoutNaN);

    %选取正特征
    if posLenth < rankPLength
        rankPLength = posLenth;
    end

    selectedFeaturesMatrix{1,columIndex}=pos(:,1:rankPLength);

    if rankNLength ~= 0
        %选取负特征
        selectedFeaturesMatrix{2,columIndex}=pos(:, rowNum-rankNLength:rowNum);
    end


   

%     %权重向量
%     weight_P=[];
%     weight_N=[];
% 
%     %计算每个所选正负特征的权重(当这个特征存在时分类正确的比重)
%     for index = 1:rankLength
% 
%         countP=0;
%         countN=0;
% 
%         %获得正特征序号
%         featureP = selectedFeaturesMatrix{1,columIndex}(index,1);
%         %获得负特征序号
%         featureN = selectedFeaturesMatrix{2,columIndex}(index,1);
% 
%         %遍历数据集和label寻找分类正确的个数
%         for rowIndex = 1: dataRow
% 
%             if DataSet(rowIndex, featureP) == 1 && label(rowIndex, 1) == columIndex
%                 countP = countP + 1;
%             end
% 
%             if DataSet(rowIndex, featureN) == 1 && label(rowIndex, 1) ~= columIndex
%                 countN = countN + 1; 
%             end
% 
%         end
% 
%         weight_P(index, 1) = countP / dataRow;
%         weight_N(index, 1) = countN / dataRow;
% 
%     end
% 
%     weightMatrix{1, columIndex} = weight_P;
%     weightMatrix{2, columIndex} = weight_N;


end

Mytime=toc(start);

PNcoe{1,1} = kendallMatrix + 1;



end

