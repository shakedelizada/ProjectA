function [gaborImg, labeledImg,C]  = textureExtractionCentroid(imgName, gaborArray,kmeans_dept) 
img = imread(imgName);
imageSize = size(img);
x=imageSize(1);
y=imageSize(2);

if (mod(x,4) ~= 0)
    x = x - mod(x,4);
end

if (mod(y,4) ~= 0)
    y = y - mod(y,4);
end

img = imresize(img , [x y]);
original = img;
img=rgb2gray(img);

imageSize = size(img);
x=imageSize(1);
y=imageSize(2);
numRows = ((x-(mod(x,4))) / 4);
numCols = ((y-(mod(y,4))) / 4);

%gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
featureVector = gaborFeatures(img,gaborArray,4,4);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.
gaborMatNoIndex = reshape(featureVector,numRows*numCols,[]);
X = 1:numRows;
Y = 1:numCols;
[X_mat,Y_mat] = meshgrid(X,Y);
X_vec = reshape(X_mat,[],1);
Y_vec = reshape(Y_mat,[],1);
gaborMat = [X_vec  Y_vec gaborMatNoIndex];
% 
% gaborMat = bsxfun (@minus,gaborMat, mean(gaborMat));
% gaborMat = bsxfun(@rdivide,gaborMat,std(gaborMat));
        
% %transform each row to norm = 1
pixelNum = numRows*numCols;
for j = 1:pixelNum
    gaborMatNoIndex(j,:) =  gaborMatNoIndex(j,:)./norm( gaborMatNoIndex(j,:),40); 
end

% coeff = pca(gaborMat);
% feature2DImage = reshape (gaborMat*coeff(:,1),numRows,numCols);
% figure
% imshow(feature2DImage,[])

rng(1);

[L,C] = kmeans(gaborMatNoIndex,kmeans_dept,'Replicates',5);
l_size=size(L);
L = reshape(L,[numRows numCols]);
% figure
% subplot(1,2,1);
% imshow(original);
% subplot(1,2,2);
% imshow(label2rgb(L));

gaborImg = gaborMatNoIndex;
labeledImg = L ;
end 