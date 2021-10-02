python train.py \
  --root-dir '../' \
  --saved_fn 'yolo3d_yolov4_im_re' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/yolo3d_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --gpu_idx 0 \
  --evaluate \
  --resume_path ../checkpoints/yolo3d_yolov4_im_re/Model_yolo3d_yolov4_im_re_epoch_215.pth 
