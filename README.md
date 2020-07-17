# Yolo-building-dataset-tool

Credit to @ManivannanMurugavel for the main.py and part of selection.py

**This tool is used to label your dataset, create data files and organize things.**

### Label your images

Run label_images.py
- Enter the folder name that contains your images (relative to the Images/ folder)
- Label your images and navigate through them using "<< Prev" and "Next >>" or with the keyboard 'a' and 'd'

### Label directly from videos with possibility to run detection with darknet

This tool can be used to make the frame selection and labeling task easier.

With this tool you can use a pretrained YOLO model to help you to complete a dataset by looking at what would be the output of the algorithm.
```
usage: selection_labeling.py [-h] [--yolo] [--darknet-path DARKNET_PATH]
                             [--config CONFIG] [--weight WEIGHT] [--meta META]

Tool to select and label frames from videos

optional arguments:  
  -h, --help            show this help message and exit  
  --yolo                Indicate if you want to use existing YOLO model to
  			pre-label frames. Not enable by default  
  --darknet-path DARKNET_PATH 
  			Path to the darknet repo  
  --config CONFIG       Path of the yolo config file  
  --weight WEIGHT       Path of the yolo weights file  
  --meta META           Path of the yolo meta file  
```
The --yolo argument allows to enable or not the pre-labeling with an existing model.

![GUI of the selection/labeling tool](yolo_annotation_tool.png?raw=true "YOLO annotation tool")

When saving a frame (hit button Save or S on the keyboard) both the frame and the labels (in YOLO format) will be save in the Output folder.

### Process your dataset to make it ready for YOLO training

Run process.py 
- python process.py --dataset 'Images/myImages' --target-path 'myDataset' --nbFolds 2
	- dataset is the path to your images relative to the executing file
	- target-path is the path were you want to store the dataset relative to the darknet folder
	- nbFolds argument can be changed to do a K-Folds cross-validation split of the data
	
NOTE: Please create or update classes.txt file and write all classes that you train for. Also the 'obj.names' will be created by 'process.py' based on this txt file so please double check that it is correct for your dataset.

The dataset is ready for yolo training. You can now move your dataset folder to the darknet directory.

