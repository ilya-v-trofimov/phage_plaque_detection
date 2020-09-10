# Phage growth analysis tool
This tool takes Petri dish images as input and analyses the parameters of the phage blocks and clusters

To run the tool, follow the steps below:
## Prerequisites
Python 3.6 must be installed in the system
## Install dependencies
Run following command in the root of the project:
```
pip install -r requirements.txt
``` 
## Run the tool
To run the tool, use one of the following options. Output will be placed into `./out` directory of the project 
### on a directory
If it's required to analyze multiple files, put all files in the directory. Then run the following command:
```
python contours_detector.py -d <path_to_the_directory>
```

### on a file
 If it's required to analyze a single file, run the following command:
```
python contours_detector.py -i <path_to_the_file>
```
