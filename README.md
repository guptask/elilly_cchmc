#Confocal images analysis for study of Eli Lilly Images

##Packages to install

+ ###opencv: 
>Clone the opencv repo at https://github.com/Itseez/opencv. Follow the 
instructions for installation on Linux at http://opencv.org/. Add to 
~/.bashrc **export LD\_LIBRARY\_PATH=${LD\_LIBRARY\_PATH}:/usr/local/lib**. 
To test sample opencv code, compile using 
**g++ <file\_name> `pkg-config opencv --cflags --libs`**


##Build and run microglia analysis package

+ Inside the project root directory, type **make** to build the project.
A binary called **analyze** will be created.

+ Command to run the software: 
**./analyze <image directory path with / at end>**

+ Image directory path should have a **tiff** directory which contains the 
tiff image directories for each czi image.

+ **image_list.dat** has to be created inside the image directory path. This 
tracks the different czi images that are being processed and allows selective 
processing of one or more images.

##Result

+ Inside the image directory path, a directory called **result** gets created. 
This contains the raw, enhanced and analyzed images for each image.

+ The **computed_metrics.csv** contains the metrics results generated during 
the analysis.

