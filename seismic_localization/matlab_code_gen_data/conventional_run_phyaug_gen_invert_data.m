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

src_opt = [25, 50, 100, 200, 500, 1000, 2000, 3000,4000,5000,6000,7000,8000];
noise_opt = [1,2,3,4,5,6,7,8,9,10];
noise_factor = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
% noise_opt = [2]
% noise_factor = [0.02]


for j = 1:length(noise_opt)
    for i = 1:length(src_opt)
        %gen ray path
        NoSrc = src_opt(i); % how many true event source
        num = src_opt(i); % how many true event source
        NPCT = noise_opt(j) % noise percent in integer
        NFCT = noise_factor(j) % noise percent in float
        %gen csv file, used in gen_ray_path_script_fix_Rec
        GenCsv = false;
        % user inverted slowness map
        use_invert = false;
        gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
        conventional_tomo_phyaug; % code to derive slowness map using conventional method

        %gen ray path
        NoSrc = 1000;
        
        %gen csv file, used in gen_ray_path_script_fix_Rec
        GenCsv = true;
        % user inverted slowness map
        use_invert = true;
        for z = 1:30
            if use_invert == true
                fname = sprintf('model_%s_src_%d_rec_8_data_%d_%dpct_noise_inverted_with_%d_src.csv',map,NoSrc,z,NPCT,num);
            else
            fname = sprintf('model_%s_src_%d_rec_8_data_%d_true_%dpct_noise.csv',map,NoSrc,z,NPCT)
            end
            gen_ray_path_script_fix_Rec; % code to configure sensor array, and to calculate travel times assuming straight-rays
        end
    end
end
