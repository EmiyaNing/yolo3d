## Now 2021\9\27
1. Install shapely.
2. Install opencv==4.2.0.30
3. Install mayavi.
> In this process, I have met many erros.
> 
> First, when i install the mayavi by pip, the install icon will stucking.
> 
> The problem is occred by vtk's version don't match mayavi's needed version.
> 
> The mayavi 4.7.3 need vtk's version is 9.0.1,but the pip will install vtk 9.0.3.

4. Run the kitti_dataloader.py file, and visualize BEV maps and camera images.
5. Run the train.py, and waiting for result.

## Now 2021\10\2
1. Training a yolo3d model.It's result is higher 10% map in BEV than original yolo3d-yolov2
2. Testing the yolo3d-yolov4-epoch220.pth, and save the result in a video
3. Add a __str__ function in data_process.kitti_data_utils.object3d
4. Add a test_official.py file
5. Add a test_official.sh file

#### Next step
1. Fix the model's outpu to let yolo3d can predict whether the object is occlusion or trunction
2. Finish the test_official.py file to save the model's predict result in a txt file
3. Test the yolo3d each output class's easy moderate hard result.
4. Try to draw a attention map for yolo3d-yolov4