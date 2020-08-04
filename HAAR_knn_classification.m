clc;        
clear;
close all;

%% Variable declarations

[a, b]= uigetfile('*.*');
b=[b a];
I=imread(b);
clear a;
clear b;

Objects=ObjectDetection(I,'haarcascade_frontalface_alt.mat');
Num_rows=size(Objects);

if Num_rows>1
    final_Object=sum(Objects)/(Num_rows(1));
else final_Object=Objects;
end

ShowDetectionResult(I,final_Object);

figure
im=imcrop(I,final_Object);
%  im=imresize(I,[256 256]);
imshow(im);

%% NOSE DETECTION:
%To detect Nose
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',10);
BB1=step(NoseDetect,im);
% figure,
% imshow(im); hold on
for i = 1:size(BB1,1)
    rectangle('Position',BB1(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end
title('Nose Detection');
hold on;

%% Eye Pair DETECTION:
%To detect Eye Pair
EyePairDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',16);

BB2=step(EyePairDetect,im);
r=size(BB2);

if(r(1,1)==0)
    EyePairDetect = vision.CascadeObjectDetector('EyePairSmall','MergeThreshold',1);
end
BB2=step(EyePairDetect,im);

% figure,
% imshow(im); hold on

for i = 1:size(BB2,1)
    rectangle('Position',BB2(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end
title('Eye pair Detection');
hold on;


%% Mouth DETECTION
%To detect Mouth
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',20);
BB3=step(MouthDetect,im);
% figure,
% imshow(im); hold on
for i = 1:size(BB3,1)
    rectangle('Position',BB3(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end
title('Mouth Detection');
hold on;



%% Face profile DETECTION:
%To detect Mouth
FaceProfileDetect = vision.CascadeObjectDetector('ProfileFace','MergeThreshold',10);
BB4=step(FaceProfileDetect,im);
% figure,
% imshow(im); hold on
for i = 1:size(BB4,1)
    rectangle('Position',BB4(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
end
title('FaceProfile Detection');
hold on;


%% Preprocessing of image

im2=rgb2gray(im);
imshow(im2);
figure;

im2 = imadjust(im2,stretchlim(im2),[]);
imshow(im2);
figure;

im2=histeq(im2);
imshow(im2);

%im2=medfilt2(im2,[3 3]);
im2 = filter2(fspecial('average',3),im2)/255;


[Gmag, Gdir]=imgradient(im2,'sobel');
figure, imshow(Gmag)

im3=im2bw(Gmag,graythresh(Gmag));
figure,imshow(im3);




im2A=imcrop(Gmag,[BB2(1,1) final_Object(1,2) BB2(1,3) -final_Object(1,2)+BB2(1,2)]);
im2B=imcrop(Gmag,[BB2(1,1) BB1(1,2) -BB2(1,1)+BB1(1,1) BB1(1,4)]);
im2C=imcrop(Gmag,[BB1(1,1)+BB1(1,3) BB1(1,2) -BB2(1,1)+BB1(1,1) BB1(1,4)]);
im2D=imcrop(Gmag,[final_Object(1,1)-(-BB2(1,1)+final_Object(1,1))*2/3 BB2(1,2) (BB2(1,1)-final_Object(1,1))*2/3 BB2(1,4)]);
im2E=imcrop(Gmag,[BB2(1,1)+BB2(1,3) BB2(1,2) (BB2(1,1)-final_Object(1,1))*2/3 BB2(1,4)]);


im2A=double(im2A);
im2B=double(im2B);
im2C=double(im2C);
im2D=double(im2D);
im2E=double(im2E);
im4=~im3;


figure,imshow(im4);

im3=double(im3);
im3A=imcrop(im3,[BB2(1,1) final_Object(1,2) BB2(1,3) -final_Object(1,2)+BB2(1,2)]);
im3B=imcrop(im3,[BB2(1,1) BB1(1,2) -BB2(1,1)+BB1(1,1) BB1(1,4)]);
im3C=imcrop(im3,[BB1(1,1)+BB1(1,3) BB1(1,2) -BB2(1,1)+BB1(1,1) BB1(1,4)]);
im3D=imcrop(im3,[final_Object(1,1)-(-BB2(1,1)+final_Object(1,1))*2/3 BB2(1,2) (BB2(1,1)-final_Object(1,1))*2/3 BB2(1,4)]);
im3E=imcrop(im3,[BB2(1,1)+BB2(1,3) BB2(1,2) (BB2(1,1)-final_Object(1,1))*2/3 BB2(1,4)]);



figure
imshow(im3A)
figure
imshow(im3B)
figure
imshow(im3C)
figure
imshow(im3D)
figure
imshow(im3E)


im3A=double(im3A);
im3B=double(im3B);
im3C=double(im3C);
im3D=double(im3D);
im3E=double(im3E);

v1 =[-BB3(1,1)-BB3(1,3)/2+BB2(1,1), -BB3(1,2)-BB3(1,4)+BB2(1,2)];
v3 = [BB3(1,1)+BB3(1,3)/2-BB2(1,1)+BB2(1,3), BB3(1,2)+BB3(1,4)-BB2(1,2)+BB2(1,4)];

u1 = v1 / norm(v1);
u3 = v3 / norm(v3);


W1=(sum(sum(im3A)));
M1=(sum(sum(im2A)));
P1=length(im3A)^2;

W2=(sum(sum(im3B)));
M2=(sum(sum(im2B)));
P2=length(im3B)^2;

W3=(sum(sum(im3C)));
M3=(sum(sum(im2C)));
P3=length(im3C)^2;

W4=(sum(sum(im3D)));
M4=(sum(sum(im2D)));
P4=length(im3D)^2;

W5=(sum(sum(im3E)));
M5=(sum(sum(im2E)));
P5=length(im3E)^2;

W=W1+W2+W3+W4+W5;
P=P1+P2+P3+P4+P5;
M=M1+M2+M3+M4+M5;

R1=BB2(3);
R2=pdist([BB2(1:2);BB3(1:2)],'euclidean');
R3=pdist([BB2(1:2);BB1(1:2)],'euclidean');
R4=pdist([BB2(1:2);final_Object(1,1)+final_Object(1,4) final_Object(1,1)+(final_Object(1,3))/2],'euclidean');

F1=R1/R3;
F2=R1/R2;
F3=R3/R4;
F4=R3/R2;

F5= (W)/(P);
F6=(M)/(255*abs(W));
F7=M/(255*P);
F8 = acos(dot(u1, u3));
fea1=[F1,F2,F3,F4,F5,F6,F7,F8];


%% KNN Classifications

load haar_features  % fea and group

train_set_1=fea(:,1:4);

test_set_1=[F1 F2 F3 F4];

class=group';

result=knnclassify(test_set_1,train_set_1,class);


if result == 2
    msgbox('Child')
else
    
    train_set_2=fea(:,5:8);
    test_set_2=[F5 F6 F7 F8];
    class=group';
    
        result = knnclassify(test_set_2,train_set_2,class);

    
    if result ==1
        msgbox('Adult')
    elseif result == 3
        msgbox('senior')
    end
end

