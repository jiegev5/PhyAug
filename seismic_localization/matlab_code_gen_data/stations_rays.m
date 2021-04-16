%============================================================
% Code for configuring sensor array, and calculating rays and travel times
% between sensors (straight-ray code) for locally sparse travel time 
% tomography (LST) algorithm simulation. The methods implemented here are 
% the same as described in:
% 1. M.J. Bianco and P. Gerstoft, "Travel time tomography with adaptive 
% dictionaries," IEEE Trans. on Computational Imaging, Vol. 4, No. 4, 2018.
% 
% If you find this code useful in your research, please cite the above
% reference (1).
%
% LST version 1.0
% Michael J. Bianco, December 11th, 2018.
% email: mbianco@ucsd.edu
%============================================================

nStations = 64; % number of stations 

% generating random array locations
rng(0) %make sure each time give same loc
stas = randn(nStations,2); %64 by 2 array
stas = sign(stas).*abs(stas).^(7/8); % make center less dense
staRatio = 49./max(abs(stas));
xSta = stas(:,1)*staRatio(1)+49;
ySta = stas(:,2)*staRatio(2)+49;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

staPerms = nchoosek(1:nStations,2)'; % nchoosek pairs
lenStaPerms = length(staPerms); %number of pairs, 2016

staPermsVec = staPerms(:); %1-D vector, 4032

xStaChoose = xSta(staPermsVec); % map 4032 coordinates to xStaChoose
%xSta contains X coordinates, staPermsVec contains sequence, 1 2, 1 3....
yStaChoose = ySta(staPermsVec);

xStaResh = reshape(xStaChoose,[2,lenStaPerms]);
yStaResh = reshape(yStaChoose,[2,lenStaPerms]);

% each X1, X2 refer to a source-receiver pair and can form a straight line
X1 = [xStaResh(1,:);yStaResh(1,:)]'; %can refer to source
X2 = [xStaResh(2,:);yStaResh(2,:)]'; %can refer to station

%% determining valid(sampled) region

% bps = ginput;
bps= [ 7.7189   39.5190;
   16.2442   21.4140;
   31.6820   14.0671;
   45.7373   11.7055;
   70.8525   16.6910;
   91.3594   22.9883;
   97.8111   44.2420;
   88.1336   66.5452;
   70.8525   77.5656;
   36.5207   97.7697;
   28.6866   89.6356];   
   
vb = roipoly(100,100,bps(:,1),bps(:,2)); % valid bounds corresponding to outermost ray paths

se = strel('square',9);
vb2 = imdilate(vb,se); % valid bounds for LST


%% calculating rays and travel travel time and sensing matrix
display('Solving for ray paths and travel time...')

mult = 100; % ensuring enough resolution for ray steps (can increase this number to improve accuracy)
sc=size(sTrue);
lc=length(sTrue(:));
steps = ceil(1.5*max(sc)); % step size for ray propagation


av=single([]); % used to store ray length in each pixel
ii=single([]); % used to store identify ray number and ray's pixel count
jj=single([]); % used to store pixel loc for each ray
for m = 1:length(X1)
    src = X1(m,:)';
    rec = X2(m,:)';
    % finding ray trajectory
    v0 = rec-src;
    vn = norm(v0);
    v = v0/vn; %ray vector (normalized)
    
    % tracing ray
    points = src+v*linspace(0,steps,steps*mult); %stepping through ray trajectory, number of points should be large, includes endpoints
    pcheck = sum((points-rec).^2); % finding intersection of ray with pixel
    [u,i]=min(pcheck); % u is length, i is indexing
    inds = floor(points(:,2:i)); % starts @ 2 since ray travels from source to receiver
    % use floor to set points in same pixel to same loc
    npoints=length(inds); %how many points for this ray

    xs=inds(1,:); 
    ys=inds(2,:);
    
    ds_int = (vn/npoints); % length of ray/#points
          
    indA = sub2ind(sc,ys',xs'); % linear indexing for each point in 100x100 matrics
    
    a0=zeros(1,lc); % generate a 100x100 0 matrix
    for k = 1:length(xs)
        a0(indA(k))=a0(indA(k))+ds_int; % % calculate ray path in each cell/pixel
    end
    
    colInds=unique(indA); % unique indices for each ray
    % av store the length of ray path in each pixel
    av=[av;a0(colInds)']; % tomo mtx vector, much faster than indexing array
    % ii used to identify ray number (which ray is this)
    ii=[ii;m*ones(length(colInds),1)];
    % jj used to store each ray pixel loc in linear indexing
    jj=[jj;colInds];

end
% generate a 2016x10000 matrix A, for each raw represent the linear pixel loc
% and ray path leagth in each pixel
A=sparse(double(ii),double(jj),double(av),length(X1),length(sTrue(:)));
A=full(A);
% obtain arrival time
Tarr = A*sTrue(:);

%% plotting slowness map and rays

figure(1)
clf                 %  edge gap   vertical    horizontal
ha=tight_subplot(1,3,[0.01 0.02],[.22 .22],[.095 .05]);

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

bigTitle=['Slowness map, stations and rays, ray density'];
axes(ha(1))
text(80,-30,bigTitle,'fontsize',16,'interpreter','latex')





