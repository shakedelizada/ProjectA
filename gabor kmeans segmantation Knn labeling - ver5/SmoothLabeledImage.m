function [] = SmoothLabeledImage(img)


img = imread(img);

[rows,cols] = size(img);

bw_img = zeros(rows,cols);


for i = 1:3
    bw_img(img == i) = 1;
    img_no_small_patches = bwareaopen(bw_img,5);
    figure();
    imshow(img_no_small_patches);
end
    







end

