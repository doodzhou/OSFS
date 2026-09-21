function [selectedFeatures, time] = SOGSFS(X, Y111,  alpha, testData, testLabels,bb)
    % OSGFS_FI_interr - online streaming feature selection considering feature interaction
   options = [2, 1000, 1e-5, 0];  
  
    unique_labels = unique(Y111);


    label_to_index = containers.Map(unique_labels, 1:length(unique_labels));


    Y1 = arrayfun(@(x) label_to_index(x), Y111);
    start = tic;
    [d, P] = size(X);
    mode = zeros(1, P);
    t = 0;
    hist = [];
    class_num=length(unique(Y1));
%       G = floor(P / 10);
     G=100;
    h_p=[];
    Y_hist=[];
    W_hist=[];
    M_hist=[];
    numLabeled = ceil(bb * d);      
    labeledIndices = randperm(d, numLabeled);  
    labeledIndices=sort(labeledIndices)
   
    Y_labeled = Y1(labeledIndices);
    for i = 1:G:P
        disp(i);  
        disp(P);  

        i_end = G + i - 1;
        G1 = G;
        
        if i_end > P
            i_end = P;
            G1 = P - t * G;
            break;
        end
        
        if t == 0
            indexArray = [find(mode == 1), i:i_end];
        else
            indexArray = [find(mode == 1), i:i_end];
%             indexArray = union(hist, indexArray1);
        end
        
        X_G = X(:, indexArray);
        [U2, c] = sfcm(X_G, class_num, labeledIndices, Y_labeled , 2,1000,0.00001,5,0);
        U2=U2';
        if t~=0
             alpha1 = 0.1; 
             U2 = alpha1 * M_hist + (1 - alpha1) * U2;
        end
        [~, m1] = max(U2, [], 2);

     
        mic=[]
        for i=1:G1
            mic1=MIC_an(X_G(:,i)',m1');  
            mic=[mic,mic1];
        end
        mean1=mean(mic);
        std1=std(mic);
%         th=mean1+0.5*std1;
        th=mean1+std1;
        indices = find(mic > th);
        indices = sort(indices);


        X_G1 = X_G(:, indices); 
        [B, FitInfo] = lasso(X_G1, m1,'Alpha', 1, 'CV', 10); 
        minInd=FitInfo.IndexMinMSE;
        SF=B(:,minInd);
        selectedFeatures=indexArray(indices(SF~=0));
        hist = [hist,selectedFeatures];  

        M_hist=U2;
        t = t + 1;
    end

%     final features
    X_G = X(:, hist);
    [~, m1] = max(M_hist, [], 2);
    [B, FitInfo] = lasso(X_G, m1,'Alpha', 0.5,  'CV', 10);
     minInd=FitInfo.IndexMinMSE;
     SF=B(:,minInd);
     selectedFeatures=hist(SF~=0);
    
    time = toc(start);
end

function[mic_single]=MIC_an(data,Y)

    dep = mine(data,Y);
    mic_single=dep.mic;

end

function[mic_set]=MIC_Set(data,Y)

[n,p]=size(data);
single_mic=0;
each_mic=0;

for i=1:n-1
    each_mic=MIC_an(data(i,:),data(n,:))+each_mic;
    single_mic=MIC_an(data(n,:),Y);
   
end
if n==1
    mic_set=0;
else
    mic_set=single_mic-each_mic/(n-1);
end
end

function [center, U, obj_fcn] = semi_fcm(data, cluster_n, options, labeledIndices, Y_labeled)
%SEMI_FCM Semi-supervised fuzzy c-means clustering.
%
%   [CENTER, U, OBJ_FCN] = SEMI_FCM(DATA, N_CLUSTER, OPTIONS, LABELEDINDICES, Y_LABELED)
%   performs semi-supervised fuzzy c-means clustering, where labeledIndices and
%   Y_labeled specify a set of labeled data points and their associated labels.
%
%   The function adds label constraints into the clustering process.
%
%   Example:
%       data = rand(100, 2);
%       labeledIndices = [1, 2, 3, 10, 20];  % Labeled data points
%       Y_labeled = [1, 1, 2, 2, 1];  % Labels corresponding to the labeled data
%       [center, U, obj_fcn] = semi_fcm(data, 2, [], labeledIndices, Y_labeled);

if nargin ~= 2 && nargin ~= 3 && nargin ~= 5
    error('Incorrect number of input arguments.');
end

data_n = size(data, 1);

% Default options
default_options = [2;	% exponent for the partition matrix U
                   100;	% max. number of iterations
                   1e-5;	% min. amount of improvement
                   1];	% info display during iteration 

if nargin < 3
    options = default_options;
else
    % If "options" is not fully specified, pad it with default values.
    if length(options) < 4
        tmp = default_options;
        tmp(1:length(options)) = options;
        options = tmp;
    end
    % If some entries of "options" are nan's, replace them with defaults.
    nan_index = find(isnan(options) == 1);
    options(nan_index) = default_options(nan_index);
    if options(1) <= 1
        error('Exponent for the partition matrix U must be greater than 1.');
    end
end

expo = options(1);		% Exponent for U
max_iter = options(2);	% Max. iteration
min_impro = options(3);	% Min. improvement
display = options(4);	% Display info or not

obj_fcn = zeros(max_iter, 1);	% Array for objective function

% Initialize fuzzy partition
U = initfcm(cluster_n, data_n);			

% Main loop
for i = 1:max_iter
    % Step update of fuzzy membership and centers
    [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
    
    % Incorporate semi-supervised information (hard constraint for labeled points)
    for j = 1:length(labeledIndices)
        idx = labeledIndices(j);
        label = Y_labeled(j);
        % Hard assign labeled point to the correct cluster
        U(:, idx) = 0;  % Reset membership
        U(label, idx) = 1;  % Set full membership in the corresponding cluster
    end
    
    % Recalculate the centers using the modified membership
    for k = 1:cluster_n
        % Ensure proper dimensionality for matrix multiplication
        U_k = (U(k, :) .^ expo)';  % Make U_k a column vector
        center(k, :) = (U_k' * data) / sum(U_k);  % Calculate new cluster center
    end

    if display
        fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
    end

    % Check termination condition
    if i > 1
        if abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro, break; end
    end
end

iter_n = i;	% Actual number of iterations 
obj_fcn(iter_n+1:max_iter) = [];
end

function [U, center, obj_fcn] = sfcm(data, cluster_n, labeledIndices, Y_labeled, m, max_iter, e, alpha, printOn)
   
    if nargin < 8
        printOn = 1;
    end
    if nargin < 7
        alpha = 5;
    end
    if nargin < 6
        e = 0.00001;
    end
    if nargin < 5
        max_iter = 1000;
    end
    if nargin < 4
        m = 2;
    end
    data_n = size(data, 1);
    U = initfcm(cluster_n, data_n);	
    
%     [U, center, obj_fcn(i)] = stepfcm(data, U, cluster_n, expo);
    F=zeros(cluster_n,data_n);
    for j = 1:length(labeledIndices)
        idx = labeledIndices(j);
        label = Y_labeled(j);
        % Hard assign labeled point to the correct cluster
        F(:, idx) = 0;  % Reset membership
        F(label, idx) = 1;  % Set full membership in the corresponding cluster
    end
 
    % 
    for i = 1:max_iter
        % Step update of fuzzy membership and centers 
        [U, center, obj_fcn(i)] = stepfcm1(data, U, cluster_n, m, F, alpha);
      
        if i > 1
            if abs(obj_fcn(i) - obj_fcn(i-1)) < e
                break;
            end
        end
    end
end

function [U, center, obj_fcn] = stepfcm1(data, U, cluster_n, m, F, alpha)
    [U_fcm, center, obj_fcn] = stepfcm(data, U, cluster_n, 2);
    dist = distfcm(center, data); 
    a=(repmat(sum(F, 1), cluster_n, 1));
    U_2 = (alpha / (alpha + 1)) * U_fcm .*a;

    U = U_fcm + (alpha / (1 + alpha)) * F - U_2;
 
    obj_fcn = sum((dist.^2) * U.^m) + alpha * sum((dist.^2) * (U-F) .^ m);
    obj_fcn=sum(obj_fcn);
end

function center = centercompute(data, U)
    cluster_n = size(U, 1);  
    mf = zeros(1, cluster_n);  
    
    for i = 1:cluster_n
        mf(1, i) = sum(U(i, :));  
    end
    
   
    mf = repmat(mf, size(data, 2), 1);  
    mf = mf';  
   
    center = (U * data) ./ mf;  
end

function dist = distfcm(data, center)
   
    [num_samples, ~] = size(data);  
    [num_centers, ~] = size(center); 
    
    dist = zeros(num_centers, num_samples);  
    
    
    for i = 1:num_centers
        for j = 1:num_samples
            dist(i, j) = sqrt(sum((data(j, :) - center(i, :)).^2));  
        end
    end
end

function new_x = tmp(x)
    [n, m] = size(x);  
    new_x = zeros(n, m);  

    for i = 1:n  
        for j = 1:m  
            new_x(i, j) = x(i, j) / sum(x(:, j)); 
        end
    end
end

function D_KL = KLD_Cal(data,y)
    Var1 = var(data);
    Var2 = var(y);

    P1 = corrcoef(data',y');
    P = min(min(P1));
    Sim = Var1 + Var2 - sqrt((Var1 + Var2)^2 - 4 * Var1 * Var2 * (1 - P^2));
    D_KL = Sim / (Var1 + Var2);
end

