function [] = KnnLabelingByGaborFeatures()


%========== Data Tables for Knn =================:

[urban_avg_data_mat,urban_label_vec] = calcAvgGaborPixelMatrix('urban', 2);%urban
[forest_avg_data_mat,forest_label_vec] = calcAvgGaborPixelMatrix('forest', 1);%rural
[agri_avg_data_mat,agri_label_vec] = calcAvgGaborPixelMatrix('agriculture', 3); %agriculture

Tbl=[urban_avg_data_mat; forest_avg_data_mat ; agri_avg_data_mat] ; %data table for building knn classifier
Labels = [urban_label_vec; forest_label_vec; agri_label_vec] ; %response data for knn classifier- the corresponding labels for each vector

Mdl = fitcknn(Tbl,Labels,'NumNeighbors',1500);

images = dir(fullfile('img',filesep)); %returns an array of struct for each file in the folder
img_num = length(images); % number of file in the folder
 
   for i = 3: img_num 
        filename = fullfile('img',images(i).name); % full path of the image 
        img = imread(filename);  
        
        imageSize = size(img);
        x=imageSize(1);
        y=imageSize(2);

        % image downsize
        x = x - mod(x,4);
        y = y - mod(y,4);

        img = imresize(img , [x y]);
        image=rgb2gray(img);

        imageSize = size(image);
        x=imageSize(1);
        y=imageSize(2);
        numRows = ((x-(mod(x,4))) / 4);
        numCols = ((y-(mod(y,4))) / 4);
        pixelNum = numRows*numCols;
        
        gaborArray = gaborFilterBank(5,8,39,39);
        currfeatureVector = gaborFeatures(image,gaborArray,4,4); % feature vector from the image 
        pixelMat = reshape(currfeatureVector,pixelNum,[]); % rows = #pixels col = #filters

%         X = 1:numRows;
%         Y = 1:numCols;
%         [X_mat,Y_mat] = meshgrid(X,Y);
%         X_vec = reshape(X_mat,[],1);
%         Y_vec = reshape(Y_mat,[],1);
%         pixelMat_idx = [X_vec  Y_vec pixelMat];

        %normalization
        %pixelAvgMat = bsxfun(@minus,pixelMat_idx, mean(pixelMat_idx)); 
       % pixelAvgMat = bsxfun(@rdivide,pixelAvgMat,std(pixelAvgMat));
        for j = 1:pixelNum
            pixelMat(j,:) =  pixelMat(j,:)./norm( pixelMat(j,:),2); 
        end

        img_label = predict(Mdl,pixelMat);
        labeled_matrix= reshape(img_label,[numRows,numCols]);
        
        mymap = [ 1 0 0 ;     %red rural 
                  0 1 0 ;     %green urban 
                  0 0 1 ] ;   % blue agri
      
        labeled_img = label2rgb(labeled_matrix ,mymap);

        figure();
        subplot(1,2,1);
        imshow(img);
        title('original image');

        subplot(1,2,2);
        imshow(labeled_img);
        title('labeled image') ; 
        
        % ============================== Smooth and Label Overlay=============================  
    [rows,cols] = size(labeledMat);
    combined = zeros(rows,cols);

    for m = 1:3 %go throw all labeles
        bw_img = ones(rows,cols);
        bw_img(labeledMat == m) = 0;
        img_no_small_patches = bwareaopen(bw_img,50,4);
%         figure();
%         imshow(img_no_small_patches);
        img_no_small_patches = ~(img_no_small_patches);
        img_no_small_patches = bwareaopen(img_no_small_patches,20,8);
%         figure();
%         imshow(img_no_small_patches);
        combined(img_no_small_patches == 1) = m;
    end
    combined(combined == 0) = labeledMat(combined == 0);
    labeled_eroded_img= label2rgb(combined ,mymap);
    figure();
    imshow(labeled_eroded_img);
    title('Clean Labeled Image');
    
    figure();
    imshow(labeloverlay(img,combined,'Colormap',mymap,'Transparency',0.7));
    title('Label Overlay Image');
        
        
   end
end