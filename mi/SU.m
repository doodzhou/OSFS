function [score] = SU(firstVector,secondVector)

hX = h(firstVector);
hY = h(secondVector);
iXY = mi(firstVector,secondVector);

score = (2 * iXY) / (hX + hY);

