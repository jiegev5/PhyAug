clc
clear all
% close all
set(0,'DefaultAxesFontSize',14)
set(0,'defaultfigurecolor',[1 1 1])

map='sd';
sTrue=slownessMap3(map); % choosing slowness map, 'ch'=checkerboard, 'sd'=smooth-discontinuous
[W1,W2] = size(sTrue);
NoRec = 10;
NoSrc = 1000;
% Class_dim = [5,5];
% =========== variables ===================

% for i = 1:1
%     fname = sprintf('model_ch_src_1000_rec_10_data_%d.csv',i)
%     gen_ray_path_script
% end

for i = 1:1000
    fname = sprintf('model_sd_src_1000_rec_10_data_%d.csv',i)
    gen_ray_path_script
end


map='ch';
sTrue=slownessMap3(map); % choosing slowness map, 'ch'=checkerboard, 'sd'=smooth-discontinuous
[W1,W2] = size(sTrue);
NoRec = 10;
NoSrc = 1000;
% Class_dim = [5,5];
% =========== variables ===================

% for i = 1:1
%     fname = sprintf('model_ch_src_1000_rec_10_data_%d.csv',i)
%     gen_ray_path_script
% end

for i = 1:1000
    fname = sprintf('model_ch_src_1000_rec_10_data_%d.csv',i)
    gen_ray_path_script
end


