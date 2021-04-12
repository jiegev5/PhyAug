# evaluation
python train_normalize_with_tf.py --type eval --input_file ../honk-models-master/google-speech-dataset.pt
python train_normalize_with_tf.py --type eval --input_file ../model/model_yesnoleftright.pt
python train_normalize_with_tf.py --type eval --input_file ../model/model_12cmd_with_tf.pt
python train_normalize_with_tf.py --type eval --input_file ../model/model_4cmd_normalize_with_tf.pt
# evaluation with 1. no tf 2. 30cm 3. 45cm
python evaluate_model.py --type eval --input_file ../model/model_12cmd_original_no_tf.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_no_tf.pt
python evaluate_model_inverse_tf.py --type eval --input_file ../model/model_12cmd_normalize_no_tf.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_original_with_tf_meetroom_loc2_30cm.pt
# with normalization
                                                                        
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_30cm.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm_full_worker16.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm_full_USBtf.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm_full_5s_combined.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm_single_ATR.pt


python evaluate_model_npy.py --type eval --input_file ../model/model_12cmd_normalize_no_tf.pt
# use feature with zscore and scaled
python evaluate_model_npy.py --type eval --input_file ../model/model_12cmd_norm_scaled_no_tf.pt
# use model trained with normalized and scaled MFCC
python evaluate_model_npy.py --type eval --input_file ../model/model_12cmd_minmax_no_tf.pt

python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_kde_meetroom_loc2_45cm_full.pt
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_kde_meetroom_loc2_45cm_clipon_v2.pt

# train model, run in /utils folder
python train_normalize_no_tf.py --type train 
# python train_normalize_with_tf.py --type train 
python train_normalize_with_meetroom_loc2_30cm.py --type train
python train_normalize_with_meetroom_loc2_45cm.py --type train
python train_normalize_with_meetroom_loc2_45cm_no_DC.py --type train
python train_normalize_with_meetroom_loc2_45cm_data_Aug.py --type train
python train_normalize_with_meetroom_loc2_45cm_data_Aug_npy.py --type train

python train_normalize_with_meetroom_loc2_45cm_single.py --type train
python train_minmax_no_tf.py --type train
python save_mfcc.py --type train
python save_spectrogram.py --type train
# Oct 12, implement ADDA
python train_npy.py --type train 
python evaluate_model_npy.py --type eval --input_file ../model/model_12cmd_norm_scaled_trained_npy.pt

# submitted version
# this one is 90% accuracy on original test .wav dataset
# data set is: 
# data process:scaled only
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_no_tf.pt
# this one is 90% accuracy on original test .npy dataset
# data set is: speech_dataset_wenjie/original_testset_npy_no_norm
# data process: scaled and divide max
python evaluate_model_npy.py --type eval --input_file ../model/model_12cmd_original.pt

# PhyAug
# data set is: 
python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm.pt

python evaluate_model.py --type eval --input_file ../model/model_12cmd_normalize_with_tf_meetroom_loc2_45cm_full_april11.pt

