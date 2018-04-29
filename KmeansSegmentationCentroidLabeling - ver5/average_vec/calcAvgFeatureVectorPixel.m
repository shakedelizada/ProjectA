function [avg_vec] = calcAvgFeatureVectorPixel(folder)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

jpgFiles = dir (fullfile('boot_strapping',folder,filesep)); %returns an array of struct for each file in the folder
len = length(jpgFiles); % number of file in the folder 

feature_vec_mat = [];
avg_vec = [];
 
 for i = 3: len 
    filename = fullfile('boot_strapping',folder,jpgFiles(i).name); % full path of the image 
    img = imread (filename);
    imageSize = size(img);
    x=imageSize(1);
    y=imageSize(2);

    % image downsize
    if (mod(x,4) ~= 0)
        x = x - mod(x,4);
    end

    if (mod(y,4) ~= 0)
        y = y - mod(y,4);
    end

    img = imresize(img , [x y]);
    img=rgb2gray(img);

    imageSize = size(img);
    x=imageSize(1);
    y=imageSize(2);
    numRows = ((x-(mod(x,4))) / 4);
    numCols = ((y-(mod(y,4))) / 4);
    pixelNum = numRows*numCols;
    
    %gabor segmentation
    gaborArray = gaborFilterBank(5,8,39,39); % create vector of 40 gabor filters 
    currfeatureVector = gaborFeatures(img,gaborArray,4,4); % feature vector from the image 
    pixelMat = reshape(currfeatureVector,pixelNum,[]); % rows = #pixels col = #filters
   
    X = 1:numRows;
    Y = 1:numCols;
    [X_mat,Y_mat] = meshgrid(X,Y);
    X_vec = reshape(X_mat,[],1);
    Y_vec = reshape(Y_mat,[],1);
    pixelMat_idx = [X_vec  Y_vec pixelMat];
    
    %normalization called Feature scaling
    %pixelAvgMat = bsxfun(@minus,pixelMat_idx, mean(pixelMat_idx)); 
    %pixelAvgMat = bsxfun(@rdivide,pixelAvgMat,std(pixelAvgMat));
    
%     %transform each row to norm = 1
%     for j = 1:pixelNum
%         pixelMat(j,:) =  pixelMat(j,:)./norm( pixelMat(j,:),40); 
%     end
   
    
    feature_vec_mat = [feature_vec_mat ; pixelMat_idx];
 end
 
%calc avg vector 
%mean of each column.
avg_vec = mean(feature_vec_mat);

avg_vec = bsxfun(@minus,avg_vec, mean(avg_vec)); 
avg_vec = bsxfun(@rdivide,avg_vec,std(avg_vec));

end

