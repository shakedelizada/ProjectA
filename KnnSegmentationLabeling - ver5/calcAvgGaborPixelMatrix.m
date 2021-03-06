function [avg_data_mat,label_vec] = calcAvgGaborPixelMatrix(folder,label)

jpgFiles = dir (fullfile('boot_strapping',folder,filesep)); %returns an array of struct for each file in the folder
len = length(jpgFiles); % number of file in the folder 

label_vec = [];
avg_data_mat = [];
 
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
   
%     X = 1:numRows;
%     Y = 1:numCols;
%     [X_mat,Y_mat] = meshgrid(X,Y);
%     X_vec = reshape(X_mat,[],1);
%     Y_vec = reshape(Y_mat,[],1);
%     pixelMat_idx = [X_vec  Y_vec pixelMat];
    
    %normalization
    %pixelAvgMat = bsxfun(@minus,pixelMat_idx, mean(pixelMat_idx)); 
   % pixelAvgMat = bsxfun(@rdivide,pixelAvgMat,std(pixelAvgMat));
   for j = 1:pixelNum
      pixelMat(j,:) =  pixelMat(j,:)./norm( pixelMat(j,:),2); 
   end
    
    avg_data_mat = [avg_data_mat; pixelMat];
    
    pixels_labels = repmat( label ,[pixelNum 1]);
    label_vec = [label_vec; pixels_labels];
 end
 
end