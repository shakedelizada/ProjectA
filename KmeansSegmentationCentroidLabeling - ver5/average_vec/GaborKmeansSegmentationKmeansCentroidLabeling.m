function [] = GaborKmeansSegmentationKmeansCentroidLabeling()


% ==========================Average Feature Vectors========================

urban_avg_vec = calcAvgFeatureVectorPixel('urban');
rural_avg_vec = calcAvgFeatureVectorPixel('forest');
agri_avg_vec = calcAvgFeatureVectorPixel('agriculture');

% urban_avg_vec = calcLabelCentroid('urban');
% rural_avg_vec = calcLabelCentroid('forest');
% agri_avg_vec = calcLabelCentroid('agriculture');

% ============================Image Segmentation===========================

images = dir(fullfile('img',filesep)); %returns an array of struct for each file in the folder
img_num = length(images); % number of file in the folder
 
kmeans_dept = 10;

   for i = 3: img_num 
        filename = fullfile('img',images(i).name); % full path of the image 
        img = imread(filename);  
        imageSize = size(img);
        x=imageSize(1);
        y=imageSize(2);

        if (mod(x,4) ~= 0)
            x = x - mod(x,4);
        end

        if (mod(y,4) ~= 0)
            y = y - mod(y,4);
        end
        
        x = ((x-(mod(x,4))) / 4);
        y = ((y-(mod(y,4))) / 4);
        
        img = imresize(img , [x y]);
        
        gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
        [gaborImg,kmeansMat,Cent]=textureExtractionCentroid(filename,gaborArray,kmeans_dept);

        numRows=size(kmeansMat,1);
        numCols=size(kmeansMat,2);

        gaborImg = bsxfun (@minus,gaborImg, mean(gaborImg));
        gaborImg = bsxfun(@rdivide,gaborImg,std(gaborImg));

        meanImg =reshape (mean(gaborImg'),[numRows numCols]) ; %mean of different filters for each pixel
   
        
% ==============================Image Labeling=============================        
        
    %labeling colors
    %red = 1;
    %green = 2;
    %blue = 3;
    
    labeledMat = zeros(numRows,numCols);
    
    for j = 1 : kmeans_dept
        % auclidean distance 
        %rural 
        r_dist = rural_avg_vec-Cent(j,:);
        r_dist = sqrt(r_dist*r_dist');
        %urban
        u_dist = urban_avg_vec-Cent(j,:);
        u_dist = sqrt(u_dist*u_dist');
        %agri
        a_dist = agri_avg_vec-Cent(j,:);
        a_dist = sqrt(a_dist*a_dist');
        distance_vec = [r_dist u_dist a_dist]; 
        [~,idx] = min (distance_vec);
        if (idx == 1) %label is rural 
           labeledMat(kmeansMat == j) = 1;
        elseif (idx == 2) %label is urban 
           labeledMat(kmeansMat == j) = 2;
        else   %label is agri 
            labeledMat(kmeansMat == j) = 3;
        end
    end

    %color map 
    mymap = [ 1 0 0;       %red rural 
              0 1 0 ;      %green urban 
              0 0 1 ] ;   % blue agri
      
      
    labeled_img = label2rgb(labeledMat ,mymap);
      
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

