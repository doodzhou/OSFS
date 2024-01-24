function [SFM, PNcoe, PNRatio, acc_PN, Mytime, vote, FMeasure] = PNFS(trainData, trainLabels, testData, testLabels, labelSize, ulabel, DUM)
%PNEC 此处显示有关此函数的摘要
%   此处显示详细说明

Result_P = {};
Result_N = {};

claTotal = 3;


[PNcoe, SFM, ~, PNRatio, Mytime] = CalPN(trainData, trainLabels, labelSize, DUM);

PN_Classifier = TrainClassifier(trainData, trainLabels, SFM, ulabel, PNRatio);

[rowTest, ~] = size(testData);
[Result_P,  Result_N] = GetPNResult(rowTest, labelSize, testData, PN_Classifier, SFM, PNRatio);
[resultPN, ~, ~, ~, ~, vote] = EnsemblePN(Result_P, Result_N, rowTest, claTotal, labelSize, PNRatio);

acc_Matrix_PN = [resultPN, testLabels];
acc_PN = CalAcc(acc_Matrix_PN);


if labelSize == 2

        %二分类
        TP = length(find(resultPN == 2 & testLabels == 2));
        FP = length(find(resultPN == 2 & testLabels == 1));
        FN = length(find(resultPN == 1 & testLabels == 2));
        pre_PN = TP / (TP + FP);
        recall_PN = TP / (TP + FN);
        FMeasure = 2 * (pre_PN * recall_PN) / (pre_PN + recall_PN);

else

        %多分类
        confMat = confusionmat(testLabels, resultPN);
        % 计算每个类别的精确度和召回率
        numClasses = size(confMat, 1);
        precision = zeros(numClasses, 1);
        recall = zeros(numClasses, 1);
        fMeasure = zeros(numClasses, 1);
        
        for i = 1:numClasses

            TP = confMat(i, i);
            FP = sum(confMat(:, i)) - TP;
            FN = sum(confMat(i, :)) - TP;         
            precision(i) = TP / (TP + FP);
            recall(i) = TP / (TP + FN);            
            % 计算 F-measure
            fMeasure(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));

        end

        FMeasure = mean(fMeasure);

end


end

