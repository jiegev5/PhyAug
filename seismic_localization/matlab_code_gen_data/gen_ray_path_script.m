

% =========================================
% step 1: generate random source
% rng(0)
SrcLoc = 1 + 99*rand(NoSrc,2);
% step 2: generate random receiver loc
% RecLoc = 1 + 99*rand(NoRec,2);

RecLoc =  [4.3756   36.9532; 
           78.4427   27.4769;
           33.3973   34.3930;
           81.9302    9.6091;
           18.1883   45.7044;
           67.9357   46.0117;
           87.6844    3.8745;
           75.9584   64.0731;
           23.7329    6.8864;
           36.5354   17.7504]


% step 3: generate raypath
% generate raypath from src -> rec
LenRec = length(RecLoc);
LenSrc = length(SrcLoc);
% X1 is source points
X1 = repelem(SrcLoc,LenRec,1);
% X2 is receiver points
X2 = repmat(RecLoc,LenSrc,1);

% Step 4: solving for Arrival time
disp('Solving for ray paths and travel time wj...');

mult = 100; % ensuring enough resolution for ray steps (can increase this number to improve accuracy)
sc=size(sTrue);
lc=length(sTrue(:));
steps = ceil(1.5*max(sc)); % step size for ray propagation
av=single([]);
ii=single([]);
jj=single([]);
tic
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

toc
% generate a 2016x10000 matrix A, for each raw represent the linear pixel loc
% and ray path leagth in each pixel
% briliant!!!
% -------ray num-----Ray pixel--ray len----No of Rays---linear Slowness map
% scale_x = floor(SrcLoc(:,1)/(100/Class_dim(1))) + 1; % scale src to range(1,5)
A=sparse(double(ii),double(jj),double(av),length(X1),length(sTrue(:)));
A=full(A);
Tarr = A*sTrue(:)

B = reshape(Tarr,[LenRec,LenSrc]);
% scale_x = floor(SrcLoc(:,1)/(100/Class_dim(1))) + 1; % scale src to range(1,5)
% scale_y = floor(SrcLoc(:,2)/(100/Class_dim(2))) + 1; % scale src to range(1,5)
% number of total classes
% Class_src = sub2ind(Class_dim, scale_y,scale_x);
Result = [B',SrcLoc];
% adding header
header = {};
for i = 1:LenRec
    header = [header,['t',int2str(i)]];
end
header = [header,'src_x','src_y'];
T = array2table(Result,"VariableNames",header);
writetable(T,fname);
