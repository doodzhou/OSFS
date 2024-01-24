function [Result_P, Result_N] = GetPNResult(rowTest, labelSize, testData, PN_Classifier, SFM, PNRatio)
%GETPNRESULT 此处显示有关此函数的摘要
%此处显示详细说明

%保存测试集每个实例的正负分类结果    
for dataIndex = 1:rowTest
    
    %正分类器分类结果
    predict_knnbase_P = [];
    predict_svmbase_P = [];
    predict_treebase_P = [];
    predict_RFbase_P = [];
    %predict_C4_5_P = [];

    %负分类器分类结果
    predict_knnbase_N = [];
    predict_svmbase_N = [];
    predict_treebase_N = [];
    predict_RFbase_N = [];
    %predict_C4_5_N = [];

    %用正分类器进行分类并得到分类结果 
    for labelIndex = 1:labelSize
        predict_knnbase_P(1, labelIndex) = predict(PN_Classifier{1,labelIndex}{1,1}, testData(dataIndex, SFM{1,labelIndex}));
        predict_svmbase_P(1, labelIndex) = predict(PN_Classifier{1,labelIndex}{1,2}, testData(dataIndex, SFM{1,labelIndex}));
        predict_treebase_P(1, labelIndex) = predict(PN_Classifier{1,labelIndex}{1,3}, testData(dataIndex, SFM{1,labelIndex}));
        predict_RFbase_P(1, labelIndex) = str2num(cell2mat(predict(PN_Classifier{1,labelIndex}{1,4}, testData(dataIndex,SFM{1,labelIndex}))));
        %predict_C4_5_P(1, labelIndex) = C4_5_predict(PN_Classifier{1,labelIndex}{1,5}{1,1}, testData(dataIndex, SFM{1,labelIndex}), PN_Classifier{1,labelIndex}{1,5}{1,2})';
        %RF_Cell = predict(PN_Classifier{1,labelIndex}{1,4}, testData(dataIndex, SFM{1,labelIndex}));
        %predict_RFbase_P(1, labelIndex) =str2num(RF_Cell{1,1});
    end
    

    if PNRatio ~= 1
        %用负分类器进行分类并得到分类结果 
        for labelIndex = 1:labelSize
            predict_knnbase_N(1, labelIndex) = predict(PN_Classifier{2,labelIndex}{1,1}, testData(dataIndex,SFM{2,labelIndex}));
            predict_svmbase_N(1, labelIndex) = predict(PN_Classifier{2,labelIndex}{1,2}, testData(dataIndex,SFM{2,labelIndex}));
            predict_treebase_N(1, labelIndex) = predict(PN_Classifier{2,labelIndex}{1,3}, testData(dataIndex,SFM{2,labelIndex}));
            predict_RFbase_N(1, labelIndex) = str2num(cell2mat(predict(PN_Classifier{2,labelIndex}{1,4}, testData(dataIndex,SFM{2,labelIndex}))));
            %predict_C4_5_N(1, labelIndex) = C4_5_predict(PN_Classifier{2,labelIndex}{1,5}{1,1}, testData(dataIndex, SFM{2,labelIndex}), PN_Classifier{2,labelIndex}{1,5}{1,2})';
            %predict_RFbase_N(1, labelIndex) = predict(PN_Classifier{2,labelIndex}{1,4}, testData(dataIndex,SFM{2,labelIndex}));
        end
    end

    
    
    Result_P{dataIndex,1} = predict_knnbase_P;
    Result_P{dataIndex,2} = predict_svmbase_P;
    Result_P{dataIndex,3} = predict_treebase_P;
    Result_P{dataIndex,4} = predict_RFbase_P;
    %Result_P{dataIndex,5} = predict_C4_5_P;


   if PNRatio ~= 1
        Result_N{dataIndex,1} = predict_knnbase_N;
        Result_N{dataIndex,2} = predict_svmbase_N;
        Result_N{dataIndex,3} = predict_treebase_N;
        Result_N{dataIndex,4} = predict_RFbase_N;
        %Result_N{dataIndex,5} = predict_C4_5_N;
   else
        Result_N = {};
   end

  

end


end

