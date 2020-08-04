function Objects = ObjectDetection(image,FilenameHaarcasade,Options)


% The default Options
defaultoptions=struct('ScaleUpdate',1/1.2,'Resize',true,'Verbose',true);

% Add subfunction path to Matlab Search Path
functionname='ObjectDetection.m';
functiondir=which(functionname);
functiondir=functiondir(1:end-length(functionname));
addpath([functiondir '/SubFunctions'])

% Check inputs
if(ischar(image))
    if(~exist(image,'file'))
        error('face_detect:inputs','Image not Found');
    end
end
if(~exist(FilenameHaarcasade,'file'))
    error('face_detect:inputs','Haarcasade not Found');
end

% Process input options
if(~exist('Options','var')), Options=defaultoptions;
else
    tags = fieldnames(defaultoptions);
    for i=1:length(tags),
        if(~isfield(Options,tags{i})), Options.(tags{i})=defaultoptions.(tags{i}); end
    end
    if(length(tags)~=length(fieldnames(Options))),
        warning('image_registration:unknownoption','unknown options found');
    end
end

% Read the image from file if image is a filename
if(ischar(image))
    image = imread(image);
end

% Get the HaarCasade for the object detection
HaarCasade=Get_Haar_Casade(FilenameHaarcasade);

% Get the integral images
IntergralImages= GetIntergralImages(image,Options);

Objects = HaarCasadeObjectDetection(IntergralImages,HaarCasade,Options);

% Show the finale results
if(nargout==0)
    ShowDetectionResult(image,Objects);
end
clc;




