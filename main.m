% This is the code of ``Edge-guided Low-light Image Enhancement 
% with Inertial Bregman Alternating Linearized Minimization'', 
% Chaoyan Huang, Zhongming Wu, and Tieyong Zeng. 

% Final Retinex Egde Net with version-logplus

% By Chaoyan Huang
% Sep. 7th, 2024

clear
inputDir = './fivek/input/';
gtDir = './fivek/gt/';
edgeDir = './fivek_edge/';
resDir = './fivek_res/';
format =  '.png'; 
img = '1185'; % '0463';

gt = im2double(imread([gtDir, img, format])); % gt for psnr and ssim
exact0 = double(imread([inputDir,img, format])); % input low-light img
[m, n, c] = size(exact0(:, :, 1));
channel = 3; 
H = rgb2hsv(exact0);
S0 = H(:, :, channel);
maxS = max(max(S0));
minS = min(min(S0));
V_S = (255/(maxS-minS))*(S0-minS);

% input learned edge
g = im2double(imread([edgeDir,img, '_x1_SR', format]));
g( abs( g ) < mean(g(:))) = 0;
k=0.5;
[zz1,zz2] = gradient(rgb2gray(g));
zz1( abs( zz1 ) < mean(zz1(:)) ) = 0;
zz2( abs( zz2 ) < mean(zz2(:)) ) = 0;
zz1 = ( 1 + k * exp(  - abs( zz1 ) / 10 ) ) .* zz1;
zz2 = ( 1 + k * exp(  - abs( zz2 ) / 10 ) ) .* zz2;

%% parameters 
alpha = 23.9474; 
beta =1;
mu = 1e-5; lambda = 15; gamma = 2.2;

alpha1=0.39; beta1=0.41; 
alpha2=0.39; beta2=0.41; 

ep = 1e-2;
tau1 = ((1+ep)/ep*(2.5-2*ep+beta1)*beta)/(1-alpha1);
tau2 = ((1+ep)/ep*(2.5-2*ep+beta2)*beta)/(1-alpha2);

%convert into the logarithmic domain
s0 = log(V_S);
s = log(V_S+1);

%initialization
l = s;
r = l-s;
z = zeros(m, n);
l_old = l;
r_old = r;
%% main loop
for iter = 1:200
    tic;
    % sub-problem 1
    r = l - s;
    y1 = r + alpha1*(r-r_old);
    z1 = r+beta1*(r-r_old);
    for inner = 1:3
        r = ADMM(beta/tau1*(z1+(s-l))+y1, tau1, lambda);
        r = min(max(r, 0),log(255));
        % inner iter stop crit
        if norm(r_old-r)/norm(r_old)<1e-3
            break
        end
        r_old = r;
    end
    
    % sub-problem 2
    y2 = l+alpha2*(l-l_old);
    z2 = l+beta2*(l-l_old);
    l = FFTsolution(zz1,zz2,beta/tau2*(z2-r-s)+y2, tau2/alpha, mu/alpha);
    l = min(max(l, s),log(255));
    
    % stop crit
    crit = norm(l-l_old,'fro')/norm(l,'fro');
    if crit < 1e-4
        disp(['Outeriter: ' num2str(iter)])
        break
    end
    
    % compute time 
    iteration_times(iter) = toc;
    l_old=l;
    r_old=r;
    
    % compute energy 
    Eng(iter) = norm(Dx(r)+Dy(r))+alpha/2*norm(Dx(l)+Dy(l)-zz1).^2+beta/2*norm(l-r-s).^2;
    
    % gamma correction and find Final output
    L = exp(l);
    Ts = log(255)+(1/gamma)*(l-log(255))-r;
    S = min(exp(Ts),255);
    H(:,:,channel) = S;
    Final = hsv2rgb(H);
    
    % index computing
    PSNR(iter) = psnr(gt,Final./255);
    SSIM(iter) = ssim(gt,Final./255); 
end

cumulative_time = cumsum(iteration_times); % sum time

figure %plot curves with time in seconds 
subplot(2,2,1);plot(cumulative_time,Eng);xlabel('Time (s)');ylabel('Energy')
subplot(2,2,2);plot(cumulative_time,PSNR);xlabel('Time (s)');ylabel('PSNR')
subplot(2,2,3);plot(cumulative_time,SSIM);xlabel('Time (s)');ylabel('SSIM')
% show the Final enahnced image
subplot(2,2,4);imshow(Final/255);title(['Ours\_PSNR: ', num2str(PSNR(end)), '  SSIM: ', num2str(SSIM(end))])
% save Final enhanced image
imwrite(Final./255,[resDir, img, '_', sprintf('%2.4f',PSNR(end)),'_', sprintf('%2.4f',SSIM(end)),'.png'])


