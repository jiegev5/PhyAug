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

map='ch';
sTrue=slownessMap3(map); % choosing slowness map, 'ch'=checkerboard, 'sd'=smooth-discontinuous
[W1,W2] = size(sTrue);

%gen ray path
NoSrc = 1000;
%gen csv file, used in gen_ray_path_script_fix_Rec
GenCsv = true;
% user inverted slowness map
use_invert = true;
for i = 1:20
    if use_invert == true
        fname = sprintf('model_%s_src_%d_rec_8_data_%d_inverted_with_1000_src.csv',map,NoSrc,i);
    else
        fname = sprintf('model_%s_src_%d_rec_8_data_%d.csv',map,NoSrc,i);
    end
    gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
end
