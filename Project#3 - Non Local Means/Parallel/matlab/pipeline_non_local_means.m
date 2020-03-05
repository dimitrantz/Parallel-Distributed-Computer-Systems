%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/house.mat';
  strImgVar = 'house';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [5 5];
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  imwrite(I,'original.png');
  figure('Name','Original Image');
  imagesc(I); axis image;  
  colormap gray;
  
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );  
  imwrite(J,'noisy.png');
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  
  %% GAUSSIAN PATCH
  
  H = fspecial('gaussian',patchSize, patchSigma);  
  H = H(:) ./ max(H(:));
  H = single(H);
  
  %% USEFUL TRANSFORMATION
  
  J = padarray( J, (patchSize-1)./2, 'symmetric'); 
  J = single(J);
  
  %% PARAMETERS
  
  threadsPerBlock = [8 8];
  sizeDim = size(J);
  m = sizeDim(1);
  n = sizeDim(2);
  
  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/CudaKernel.ptx', ...
                               '../cuda/CudaKernel.cu');
  
  numberOfBlocks  = ceil( [m n] ./ threadsPerBlock );
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  
  %% DATA
  
  neib = (patchSize(1)-1)/2;
  
  G = gpuArray(J);  
  h = gpuArray(H);
  
  tic;  
  If = gather( feval(k, G, m, n, neib, filtSigma, h) ); 
  toc  
    
  %% VISUALIZE RESULT

  If = imcrop(If,[(neib+1) (neib+1) (n-1-2*neib) (m-1-2*neib)]);
  imwrite(If,'Filtered.png');
  figure('Name', 'Filtered image');
  imagesc(If); axis image;
  colormap gray;
  If= single(If);
  I= single(I);
  peakPSNR= psnr(If,I,1);
  fprintf('peakPSNR is :  %d \n',peakPSNR);
  
  J = imcrop(J,[(neib+1) (neib+1) (n-1-2*neib) (m-1-2*neib)]);
  imwrite((If-J),'Residual.png');
  figure('Name', 'Residual');
  imagesc(If-J); axis image;
  colormap gray;
  
  %% (END)

  fprintf('...end %s...\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------