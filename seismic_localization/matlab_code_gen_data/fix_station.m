nStations = 64; % number of stations 

% generating random array locations
rng(0) %make sure each time give same loc
stas = randn(nStations,2); %64 by 2 array
subplot(2,2,1); histogram(stas(:,1)); title('Random X')

stas = sign(stas).*abs(stas).^(7/8); % make center less dense
subplot(2,2,2); histogram(stas(:,1)); title('Random X with ^7/8')
staRatio = 49./max(abs(stas));
xSta = stas(:,1)*staRatio(1)+49;
ySta = stas(:,2)*staRatio(2)+49; 
subplot(2,2,3); histogram(xSta); title('Final X')
subplot(2,2,4); histogram(ySta); title('Final Y')
hold off

% step 2: generate random source
rng(0)
SrcLoc = randn(NoSrc,2)
SrcLoc = sign(SrcLoc).*abs(SrcLoc).^(7/8); % make center less dense
ratio = 49./max(abs(SrcLoc));
xSrc = SrcLoc(:,1)*ratio(1) + 49;
ySrc = SrcLoc(:,2)*ratio(2) + 49;
\


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


display('Solving for ray paths and travel time...')

mult = 100; % ensuring enough resolution for ray steps (can increase this number to improve accuracy)
sc=size(sTrue);
lc=length(sTrue(:));
steps = ceil(1.5*max(sc)); % step size for ray propagation


av=single([]);
ii=single([]);
jj=single([]);
for m = 1:length(X1)
    src = X1(m,:)'; % X1 refer to source
    rec = X2(m,:)'; % X2 refer to receiver
    % finding ray trajectory
    v0 = rec-src;
    vn = norm(v0); % length of ray path
    v = v0/vn; % ray vector (normalized)
    
    % tracing ray
    points = src+v*linspace(0,steps,steps*mult); % stepping through ray trajectory, number of points should be large, includes endpoints
    pcheck = sum((points-rec).^2); % finding intersection of ray with pixel
    [u,i]=min(pcheck); % return min distance and index
    inds = floor(points(:,2:i)); % starts @ 2 since ray travels from source to receiver
    % use floor function to choose the pixel
    npoints=length(inds);

    xs=inds(1,:); 
    ys=inds(2,:);
    
    ds_int = (vn/npoints); % length of ray/#points
          
    indA = sub2ind(sc,ys',xs'); %linear indexing for each point in 100x100 matrics
    
    a0=zeros(1,lc); % a0 marix contains length of each point interval
    for k = 1:length(xs)
        a0(indA(k))=a0(indA(k))+ds_int; % calculate ray length in each cell/pixel
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
% briliant!!!
% -------ray num-----Ray pixel--ray len----No of Rays---linear Slowness map
A=sparse(double(ii),double(jj),double(av),length(X1),length(sTrue(:)));
A(1,:)


Result = horzcat(X1,X2,Tarr)
header = {'src_x','src_y','des_x','des_y','t_arr'}
T = array2table(Result,"VariableNames",header)
filename = 'time_arr.csv'
writetable(T,filename)

xStaResh = reshape(xStaChoose,[2,lenStaPerms]);
