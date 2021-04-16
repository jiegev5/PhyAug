%============================================================
% run sequence:
% step 1: select map: sd or ch
% step 2: Set number of sources: NoSrc
%         set GenCsv: it will determine whether the data will be saved
%         set use_invert: it will use true slowness map for arrival time
%         calculation if set to false
% step 3:
%      a. set NoSrc and run with GenCsv = false, use_invert = false
%      to get slowness map sInv
%      b. set NoSrc and run with GenCsv = True, use_invert = false to get
%      ground truth data, this data will be used to validate NN model
%      c. set NoSrc and run with GenCsv = True, use_invert = True to derive
%      large amount of data for NN training

clc
% clear all
% close all
set(0,'DefaultAxesFontSize',14)
set(0,'defaultfigurecolor',[1 1 1])

map='sd';
sTrue=slownessMap3(map); % choosing slowness map, 'ch'=checkerboard, 'sd'=smooth-discontinuous
[W1,W2] = size(sTrue);

% opt = [50, 100, 200, 500, 1000, 2000, 4000];
opt = [10];
% opt = [2000];
NFCT = 0.02;


for i = 1:length(opt)
%     %gen ray path
%     NoSrc = opt(i);
      num = opt(i);
%     %gen csv file, used in gen_ray_path_script_fix_Rec
%     GenCsv = false;
%     % user inverted slowness map
%     use_invert = false;
%     
%     gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
%     conventional_tomo_phyaug; % code to derive slowness map using conventional method
    
    %gen ray path
    NoSrc = 10;
    %gen csv file, used in gen_ray_path_script_fix_Rec
    GenCsv = false;
    % user inverted slowness map
    use_invert = false;
    for j = 1:1
        fname = sprintf('model_%s_src_%d_rec_8_data_%d_2pct_noise_inverted_with_%d_src.csv',map,NoSrc,j,num);
        % fname = sprintf('model_%s_src_%d_rec_8_data_%d_inverted_with_true_2pct_noise.csv',map,NoSrc,j);
        gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
    end
end
