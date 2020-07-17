# Yolo-building-dataset-tool

Credit to @ManivannanMurugavel for the main.py and part of selection.py

**This tool is used to label your dataset, create data files and organize things.**

### Label your images

Follow these steps to prepare your dataset for YOLO:

- Run main.py
	- Enter the folder name that contains your images (relative to the Images/ folder)
	- Label your images and navigate through them using "<< Prev" and "Next >>" or with the keyboard 'a' and 'd'
- Run process.py 
	- python process.py --dataset 'Images/myImages' --target-path 'myDataset' --nbFolds 2
	- dataset is the path to your images relative to the executing file
	- target-path is the path were you want to store the dataset relative to the darknet folder
	- nbFolds argument can be changed to do a K-Folds cross-validation split of the data
	
NOTE: Please create or update classes.txt file and write all classes that you train for. Also the 'obj.names' will be created by 'process.py' based on this txt file so please double check that it is correct for your dataset.

The dataset is ready for yolo training. You can now move your dataset folder to the darknet directory.

### Label directly from videos with possibility to run detection with darknet

This tool can be used to make the frame selection and labeling task easier.

With this tool you can use a pretrained YOLO model to help you to complete a dataset by looking at what would be the output of the algorithm.


