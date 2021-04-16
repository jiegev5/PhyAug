% change noise fraction from zero to simulate noise realizations
noiseFrac = NFCT; %0.02; % noise STD as fraction of mean value of travel time (0=noise free case)
if noiseFrac == 0
    nRuns=1;
else
    nRuns=1; % number of noise realizations (calls all codes nRuns times)
end

outputs={}; % cell array for results

for nn=1:nRuns
rngSeed=nn;
% adding noise to travel time observations
stdNoise = mean(Tarr)*noiseFrac;
rng(rngSeed) % fixing random seed
rand_number = randn(length(Tarr),1);
noise = stdNoise*(rand_number/max(rand_number));
% noise = stdNoise*randn(length(Tarr),1);
Tarr_n  = Tarr+noise;

% estimating referense slowness from travel time observations
Asum = sum(A,2); %sum elements on each row, meaning path leangth for each ray
% pseuduinverse to solve linaer equation Ax=b, refer wiki
invAsum = pinv(Asum); %pseuduinverse: Asum*invAsum*Asum=Asum  invAsum = 2016x1
sRef = invAsum*Tarr_n; % scalar value of sRef

%% Running LST
% INPUTS
in_lst=[];
in_lst.lam2 = 0;             % reg. param 2

if noiseFrac==0
    in_lst.lam1 = 0;           % reg param 1
    if strcmp(map,'ch')
        Tcoeff_dlearn=1;       % number of non-zero (sparse) coefficients
        Tcoeff_nolearn=5;
    elseif strcmp(map,'sd')
        Tcoeff_dlearn=1;
        Tcoeff_nolearn=2;
    end
    
    in_lst.plots = true;       % option to plot results
else
    
    if strcmp(map,'ch')
        in_lst.lam1 = 2;          
        Tcoeff_dlearn=2;
        Tcoeff_nolearn=5;
    elseif strcmp(map,'sd')
        in_lst.lam1 = 10;           
        Tcoeff_dlearn=2;
        Tcoeff_nolearn=2;
    end
    
    in_lst.plots = false;
end

in_lst.solIter = 5;        % number of iterations of LST algorithm (set to 100 in Bianco and Gerstoft 2018 LST)
in_lst.itkmIter = 5;        % # itkm iterations
in_lst.percZeroThresh = 0.1; % threshold on the allowable percentage of unsampled pixels in patch
in_lst.noiseRealiz = 1;

in_lst.rngSeed=rngSeed; % random seed for dictionary initialization
in_lst.tomoMatrix = A;
in_lst.refSlowness = sRef;
in_lst.travelTime = Tarr_n;
in_lst.validBounds=vb2;
in_lst.normNoise = norm(noise);
in_lst.sTrue = sTrue;
in_lst.lims=[0.3 0.5];


%% Running conventional tomography code
%  (damped least squares with non-diagonal pixel covariance)

% INPUTS
in_conv=[];

if noiseFrac==0
    in_conv.eta = 0.1;       % conventional \eta regularization parameter
    in_conv.L=10;            % smoothness length scale
    in_conv.plots = true;
else
    in_conv.eta = 10;       
    in_conv.L=6;
    in_conv.plots = true;
end

in_conv.tomoMatrix = in_lst.tomoMatrix; % initialized as A
in_conv.refSlowness = in_lst.refSlowness; %sRef
in_conv.travelTime = in_lst.travelTime; %Tarr_n
in_conv.sTrue = in_lst.sTrue;
in_conv.lims=in_lst.lims;
in_conv.validBounds=in_lst.validBounds; % need to solve this
in_conv.noiseRealiz=in_lst.noiseRealiz;


% Calling conventional tomography code
% s_conv=conventional_tomo(in_conv);
eta=in_conv.eta;
L=in_conv.L;
gg=in_conv.noiseRealiz;
A =        in_conv.tomoMatrix;
sRef=      in_conv.refSlowness;
Tarr =     in_conv.travelTime;
sTrue=in_conv.sTrue;
[W1,W2]=size(sTrue);

[xxc,yyc]=meshgrid(1:W1,1:W2);
npix = W1*W2; %total number of pixels


% precalculating slowness priors (inverting pixel covariance)
disp(['Conventional: Realization #',num2str(nn),', Inverting covariance matrix'])
Sig_L=zeros(npix);
for ii=1:npix
    %%Determine the coordinates of the neighboor
    distc= sqrt((xxc-xxc(ii)).^2+(yyc-yyc(ii)).^2);
    distc = distc(:)';
    Sig_L(ii,:)=exp(-distc/L);
end
invSig_L = Sig_L\eye(npix); % A\B = INV(A)*B => A\EYE(size of A) produce a generalizaed inverse of A

%% inverting for slowness
Tref = A*(sRef*ones(npix,1)); % reference travel time

% travel time perturbation
dT=Tarr-Tref;

% inverting for slowness
disp(['Conventional: Realization #',num2str(nn),', Inverting for slowness'])
G = A'*A+eta*invSig_L;
ds = G\A'*dT;
sInv = reshape(ds,size(sTrue))+sRef
if in_conv.plots==true
    figure(2)
    clf                 %  edge gap   vertical    horizontal
    ha=tight_subplot(1,4,[0.01 0.02],[.22 .22],[.095 .05]);
    
    axes(ha(1))
    a = ha(1);
    pos = get(a,'Position');
    imagesc(sTrue,[0.3 0.5])
    h=colorbar('northoutside');
    set(a,'Position',pos) % fixing image size from colorbar addition
    xlabel(h,'Slowness (s/km)')
    xlabel('Range (km)')
    ylabel('Range (km)')
    set(gca,'Xtick',[1,20:20:100])
    set(gca,'Ytick',[20:20:100])
        
    
    axes(ha(2))
    set(gca,'Ydir','reverse')
    hold on
    for m =1:length(X1)
        plot([X1(m,1),X2(m,1)],[X1(m,2),X2(m,2)],'k')
    end
    plot(xSta,ySta,'rx','markersize',10,'linewidth',3)
    axis([1 100 1 100])
    hold off
    set(gca,'Xticklabel',[])
    set(gca,'Yticklabel',[])
    box on
    
    
    raysPerPix=reshape(sum(A~=0),size(sTrue));% calculating ray density
    
    axes(ha(3))
    a=ha(3);
    pos = get(a,'Position');
    imagesc(log10(raysPerPix))
    h=colorbar('northoutside');
    set(a,'Position',pos) % fixing image size from colorbar addition
    xlabel(h,'Rays per pixel, log_{10} ')
    axis([1 100 1 100])
    hold off
    set(gca,'Xticklabel',[])
    set(gca,'Yticklabel',[])
    box on
    
    axes(ha(4))
    a=ha(4);
    pos = get(a,'Position');
    imagesc(sInv,in_conv.lims)
    % xlabel('Range (km)')
    % ylabel('Range (km)')
    h=colorbar('northoutside');
    set(a,'Position',pos) % fixing image size from colorbar addition
    xlabel(h,'Slowness (s/km)')
    axis([1 100 1 100])
    % title('Conventional inversion example (Bayesian MAP)','fontsize',16,'interpreter','latex')

    bigTitle=['Slowness map, stations and rays, ray density and inversed map'];
    axes(ha(1))
    text(80,-30,bigTitle,'fontsize',16,'interpreter','latex')
end


end