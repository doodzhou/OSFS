function [result] = GetVote(vector)
%VOTE 此处显示有关此函数的摘要
%   此处显示详细说明
%得出vote的结果
maxVote = max(vector);
result= find(vector(1,:) == maxVote);
end

