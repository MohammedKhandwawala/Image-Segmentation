%author : Mohammed Khandwawala
%ee16b117 EE5175 Mini-Project
clear all;
close all;

%image loading
Img = imread('test3.jpg');
Img = imresize(Img,0.5);

imshow(Img);

%some parameter rtequired
[rows, cols,c] = size(Img);   
N = rows * cols;
V = reshape(Img, N, c);
%initializing weight matirx W
W = sparse(N,N);         
F = reshape(Img, N, 1, c); 
%Xs contains coordinates of the point
Xs = cat(3, repmat((1:rows)', 1, cols), repmat((1:cols), rows, 1));
Xs = reshape(Xs, N, 1, 2);

%Hyperparameters
r = 5; 
Sigma_I   = 4;                  
Sigma_X   = 5;  
segNcut = 0.7;
segArea = 1750;

%iterate on all pixel values and for each pixel values find its
for c_i=1:cols  
    for r_i=1:rows 
        %select r neighbourhood in cols
        c_j = (c_i - floor(r)) : (c_i + floor(r));
        %select r neighbourhood in rows
        r_j = ((r_i - floor(r)) :(r_i + floor(r)))';
        %check if in image dimension
        c_j = c_j(c_j >= 1 & c_j <= cols);
        r_j = r_j(r_j >= 1 & r_j <= rows);
        jN = length(c_j) * length(r_j);
        %a is the pixel index in any iteration and b are pixels in
        %neighbourhood
        a = r_i + (c_i - 1) * rows;
        b = repmat(r_j, 1, length(c_j)) + repmat((c_j -1) * rows, length(r_j), 1);
        b = reshape(b, length(c_j) * length(r_j), 1);
        X_B = Xs(b, 1, :); 
        X_A = repmat(Xs(a, 1, :), length(b), 1);
        %computing distance of A from Bs , selecting B points within r
        D_X_ab = X_A - X_B;
        D_X_ab = sum(D_X_ab .* D_X_ab, 3);
        constraint = find(sqrt(D_X_ab) <= r);
        b = b(constraint);
        D_X_ab = D_X_ab(constraint);
        %intensity of a and b points
        I_B = F(b, 1, :);
        I_A = repmat(F(a, 1, :), length(b), 1);
        Diff_I_ab = (I_A - I_B);
        Diff_I_ab = sum(Diff_I_ab .* Diff_I_ab, 3);
        %computing W
        W(a, b) = exp(-Diff_I_ab / (Sigma_I*Sigma_I)) .* exp(-D_X_ab / (Sigma_X*Sigma_X));        
    end    
end

Seg = (1:N)';

N = length(W);
d=sum(W,2);
D = spdiags(d, 0, N, N);
% computing the eigenvalue from the equation below
%'sm' argument gives eigenvalues in ascending order 
[U,S] = eigs(D-W, D, 2, 'sm'); 
U2 = U(:,2); 

%choosing the threshold that minimizes the NormCutValue
t=mean(U2);
t=fminsearch('NormcutValue',t,[],U2,W,D);

%segmenting the image based on threshold
segA = find(U2 > t);
segB = find(U2 <= t);

%recursively partitioning the partitions
[SegA  NcutA] = NormCutPartition(Seg(segA), W(segA, segA), segNcut, segArea );
[SegB  NcutB] = NormCutPartition(Seg(segB), W(segB, segB), segNcut, segArea );
Seg  = [SegA SegB];

%joining image segments 
Ncut = [NcutA NcutB];
imgArray = [];
NcutImage  = zeros(size(Img),'uint8');
for k=1:length(Seg)
 temp = 255*uint8(ones(size(V)));
 temp(Seg{k},:) = V(Seg{k},:);
 %concatenating all the segments 
 imgArray = [imgArray , reshape(temp,rows, cols,c), ones(rows, 5,c)];
end

%final Result
figure;
imshow(imgArray);%,'size',[1 NaN]);


% function to recursively compute partition
function [Seg  Ncut] = NormCutPartition(I, W, segNcut, segArea)
N = length(W);
d = sum(W, 2);
D = spdiags(d, 0, N, N); 
% calculating the second smallest eigrnvalue
[U,S] = eigs(D-W, D, 2, 'sm');
U2 = U(:, 2);

%threshold that minimizes Ncut
t = mean(U2);
t = fminsearch('NormcutValue', t, [], U2, W, D);
A = find(U2 > t);
B = find(U2 <= t);

%segArea threshold to check the segement is size is not too small and
%segncut threshold to check the min Ncut obtained is small enough 
ncut = NormcutValue(t, U2, W, D);
if ((length(A) < segArea || length(B) < segArea) || ncut > segNcut)
    Seg{1}   = I;
    Ncut{1} = ncut;     
    return;
end

%recursion
[SegA  NcutA] = NormCutPartition(I(A), W(A, A), segNcut, segArea);
[SegB  NcutB] = NormCutPartition(I(B), W(B, B), segNcut, segArea);

Seg   = [SegA SegB];
Ncut = [NcutA NcutB];
end
