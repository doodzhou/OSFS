function [PNRatio] = GetPNRatio(kendallMatrix)
%自动输出PNRatio

PNMatrix = kendallMatrix(:,1);
PElements = PNMatrix > 0;
NElements = PNMatrix < 0;
PMatrix = PNMatrix(PElements);
NMatrix = PNMatrix(NElements);
[PNUM, ~] = size(PMatrix);
[NNUM, ~] = size(NMatrix);
PNRatio = (sum(PMatrix)*PNUM) / (sum(PMatrix)*PNUM - sum(NMatrix)*NNUM);

end

