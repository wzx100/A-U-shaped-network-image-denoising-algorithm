
% clear; %clc;
addpath(genpath('./.'));
% run('E:\Tool_Box\matconvnet-1.0-beta24\matlab\vl_setupnn.m');
%% testing set
imageSets   = {'Set121','Set12','Set5','BSD68','Set68','classic5','Urban100'};
image_set   = imageSets{1};

folderTest = fullfile('testsets',image_set);

showresult  = 1;
WF = 0;
gpu = 1;

if gpu 
    gpuDevice(gpu); 
end

list_sig = [15 25 50];

ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
 

    
for Sigma = 25  %% % [ 15 25 50
    %% load model
    load('C:\Users\97913\Desktop\上传Github的代码\models\Resnet_Image200_15-epoch');
    net = dagnn.DagNN.loadobj(net) ;   
    net.removeLayer('objective') ;
    out_idx = net.getVarIndex('prediction') ;
    net.vars(net.getVarIndex('prediction')).precious = 1 ;
    net.mode = 'test';  
    if gpu
        net.move('gpu');
    end
    for i = 1 : length(filePaths)
        %%% read images
        im = imread(fullfile(folderTest,filePaths(i).name));
        im  = modcrop(im, 8);
        if size(im,3)==3
            label_im = rgb2gray(im);
            label = label_im(:,:,1);
        else
            label = im;
        end
        sz = size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label = im2single(label);


        %%
        randn('seed',0);
        
        input = label + Sigma/255*randn(sz,'single');
        
        tic;
        output = Processing_Im_w(input, net, gpu, out_idx); %%Processing_Im(input, net, gpu, out_idx);
        times(i) = toc;

        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output), 0, 0);
        if showresult
            imshow(cat(2,im2uint8(input),im2uint8(output),im2uint8(label)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
        end
        if WF 
            path =  ['./results/' modelName '/' image_set '_Sigma' num2str(Sigma)];
            if ~exist(path, 'dir'), mkdir(path) ; end
            imwrite(output, fullfile(path, [modelName '-' num2str(epoch) '-' filePaths(i).name]));
        end
        
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
    end
    if WF 
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'PSNR']), 'PSNRs');
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'SSIM']), 'SSIMs');
    end
    fprintf('PSNR / SSIM : %.02f / %0.4f, %0.4f.\n', mean(PSNRs),mean(SSIMs), mean(times));
end   
    








