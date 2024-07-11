python ./code/test.py model=CDAT comment=SHA_results_vision \
dataset=GCC_weak_strong_flip_just_weak_fix tar_dataset=SHA_weak_strong_flip_just_weak src_scale_factor=[1.0] tar_scale_factor=[1.0] tar_fix_scale_factor=1.0 src_train_crop_size=[576,768] train_crop_size=[576,768] \
search_space=[grey,scale,perspective_transform] searched_domain=[0.785,0.779,0.272] \
model_for_load_MT=./code/pre_models/SHA_92.3.pth dataroot_SHA=./data/SHA \
vision_each_epoch=300 num_workers=1