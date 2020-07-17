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
The --yolo argument allows you to enable or not the pre-labeling with an existing model.  
**WARNING:** By default the tool looks for the darknet directory at '../darknet' please specify your own path to darknet if you want this feature.

![GUI of the selection/labeling tool](yolo_annotation_tool.png?raw=true "YOLO annotation tool")

- [Prev (A)]: Look at the previous frame
- [Save (S)]: Save both the frame and the labels (in YOLO format) in the Output folder.
- [Next (D)]: Look at the next frame
- [Next10 (F)]: Look at the 10th next frame
- [Pass (G)]: Go the next video in the opened folder

You can look at the list of bounding boxes on the right. You can delete one by selecting it and press [Delete] or delete all of them with [ClearAll (X)]. Bounding boxes from the pre-trained model are automatically appearing here along with the ones you manually add.

- [ ] TODO: save frames in Images folder and labels in Label folder or change `process.py` to look in Output folder.


### Process your dataset to make it ready for YOLO training

The script `process.py` automatically create the files required by YOLO for training. Additionnally you can choose your cross-validation method. Currently you can perform holdout or kFolds.
```
usage: process.py [-h] [--dataset [/path/to/dataset]]
                  [--target-path [/path/to/target/dir]]
                  [--nbFolds [number of folds]]

Split dataset for cross-validation

optional arguments:
  -h, --help            show this help message and exit
  --dataset [/path/to/dataset]
                        Directory of the dataset relative to the executing
                        directory
  --target-path [/path/to/target/dir]
                        Directory where the data will reside, relative to
                        'darknet.exe'
  --nbFolds [number of folds]
                        Number of folds for cross-validation. If nbFolds<=2,
                        holdout cross-validation is performed with a 80/20
                        split.
```
	
**NOTE:** Please create or update classes.txt file and write all classes that you train for. Also the 'obj.names' will be created by 'process.py' based on this txt file so please double check that it is correct for your dataset.

The dataset is ready for yolo training. You can now move your dataset folder to the darknet directory.
