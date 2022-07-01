%% MACE FILTER FOR CORRELELATION
clc;
clear all;

%% loading the images

%input_dir = 'D:\m.tech bbs sem 2\ARrawdata';
%input_dir = 'D:\m.tech bbs sem 2\orl_dataset'
input_dir = 'D:\mtech bbs\acrl lab bbs\ARrawdata'
image_dim = [70,80]
filenames = dir('D:\mtech bbs\acrl lab bbs\mace_test\*.tif');
num = length(filenames)
x = []
for n = 1:num
    filename = fullfile(input_dir, filenames(n).name)
    img = imread(filename);
    %figure,imshow(img);
    img = imresize(img,[70,80])
    %img = rgb2gray(img)
    img=double(img);
    %taking 2D fft
    img=fftshift(fft2(img));
    
    %disp(img);
    %converting 2D array into 1D vector
    x(:,n)=reshape(img,[5600,1]);
end
% design of MACE filter
%computing average power spectrum
avgps=mean((abs(x).^2),2);
%computing diagonal matrix D
D=diag(avgps);
c=ones(1,num)';
cct=conj(x');
d_inv=inv(D);
xdx=inv((cct*d_inv)*x);
%computing hmace filter
h=d_inv*x*xdx*c;
%converting 1D vector into 2D array to obtain 2D correlation filter
hmace2=reshape(h,[80,70]);

%% face verification in freq. domain

I=imread('test_mace.tif');
figure;imshow(I)
I = imresize(I,[80,70])
I=double(I);
m=fftshift(fft2(I));
%k=xcorr2(m,hmace2);
k = m.*conj(hmace2)
g=ifftshift(ifft2(k));
figure;imshow(abs(g))
figure;mesh(abs(g))

%% Output in time domain

Q=imread('test_mace.tif');
%imshow(Q)
Q = imresize(Q,[80,70])

Q = double(Q)
out = xcorr2(Q,hmace2);
figure;
mesh(abs(out))
psr = PSR_calc(Q,h)
fprintf('PSR value for  Image is %f\n',psr);
colorbar

%% psr function

function PSR = PSR_calc(X_img,h_fin)
    
    out_t = xcorr2(fftshift(fft2(X_img)), conj(h_fin));
    tot_reg = abs(ifft2(fftshift(out_t)));
    figure
    %tit = strcat('Correlation Plot - ',str);
    mesh(tot_reg);% title(tit);
     
    peak = max(max(tot_reg));
    [m_peak, n_peak] = find(tot_reg == peak);
    % taking the annular region (20x20 around the peak)
    corner1 = m_peak-10;
    corner2 = m_peak+10;
    corner3 = n_peak-10;
    corner4 = n_peak+10;

    if(corner1<1)
        corner1 = 1;
    end
    if(corner2>size(tot_reg,1))
        corner2 = size(tot_reg,1);
    end
    if(corner3<1)
        corner3 = 1;
    end
    if(corner4>size(tot_reg,2))
        corner4 = size(tot_reg,2);
    end

    ann_reg = tot_reg(corner1:corner2,corner3:corner4);

    % removing peak portion from annular region (5x5 region of the peak)
    ann_reg(8:12,8:12) = 0;
    ann_reg(11,11) = peak;
    myu = mean2(ann_reg);
    st_dev = std2(ann_reg);
    PSR = (peak - myu)/(st_dev);
end

