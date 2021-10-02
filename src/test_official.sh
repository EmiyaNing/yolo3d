python test_official.py \
  --cfgfile ./config/cfg/yolo3d_yolov4.cfg \
  --batch_size 1 \
  --num_workers 1 \
  --pretrained_path ../checkpoints/yolo3d_yolov4_im_re/Model_yolo3d_yolov4_im_re_epoch_220.pth \
  --img_size 608 \
  --conf_thresh 0.6 \
  --nms_thresh 0.1 
