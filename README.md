# 💡 Light Distribution Optimization

This project aims to develop an optimization algorithm for the automated design of light distributions for camera-based object detection.

The pipeline/optimization process involves the following steps:

- Convert CSV files containing the light distribution data into PNG images using the csv_to_png function from the light_utils module.

- Run an Unreal Engine executable to generate simulation scenarios and render the images using the run_unreal_executable function from the unreal_utils module.

- Use an object detection algorithm to detect pedestrians in the rendered images and compute the detection accuracy score for each image. The default object detection algorithm is Faster R-CNN.

- Use the detection scores for each image to evaluate the light distribution and determine the optimal light distribution.

# 🛠 Requirements

- Unreal Engine (tested with version 5.1)
- Python (tested with version 3.8)
- [bdd100k-mmdet](https://github.com/SysCV/bdd100k-models/tree/main/det#install)
- [mmcv and mmdetection](https://github.com/SysCV/bdd100k-models/tree/main/det#install) 

    
# 🚀 Setup

- Install the [bdd100k-mmdet](https://github.com/SysCV/bdd100k-models/tree/main/det#install) repository.
- Following the instructions will make you install [mmcv and mmdetection](https://github.com/SysCV/bdd100k-models/tree/main/det#install) too. For the installation of mmdetection, choose the: 

    ```bash
    Case a: If you develop and run mmdet directly, install it from source.
    ```
    After installing it, go to: 
    ```bash
    /path/to/mmdetection/mmdet/apis/inference.py
    ```
    and change these instructions (line 51) :

    ```bash
    warnings.warn('Class names are not saved in the checkpoint\s meta data, use COCO classes by default.')
    model.CLASSES = get_classes('coco')
    ```
    to: 
    ```bash
    warnings.warn('Using BDD100K Classes Names')
    model.CLASSES = ["pedestrian","rider","car","truck","bus","train", "motorcycle","bicycle","traffic light", "traffic sign"]
    ```
    In some checkpoints' meta data, the classes of the bdd100k were not saved. With this modification the problem of loading false classes labels will be avoided. 

- Clone or download this repository.

- Download the trained object detection model and place it in the **"object_detection"** directory. 
(The default model used is **Faster R-CNN**, but you can use a different model by modifying the "MODEL_NAME" and "MODEL_TYPE" variables in **"detection_utils.py"**)

- Modify the **"csv_file"**, **"png_file"**, and **"exe_dir"** arguments in **"main.py"** to specify the paths for your CSV files, PNG images, and Unreal Engine executable, respectively.
# ⚡️ Usage

Activate the conda environment:
```bash
  conda activate bdd100k-mmdet
```

Run the following command to start the optimization process:
```bash
  python main.py
```

You can also specify the paths for the CSV files, PNG images, and Unreal Engine executable as command-line arguments:
```bash
  python main.py --csv_file path/to/csv_file --png_file path/to/png_file --exe_dir path/to/exe
```


# 🗃️ Files and Directories

- **"light"**: contains the CSV files and PNG images for the light distributions.

- **"object_detection"**: contains the trained object detection model and the detection code.

- **"unreal_engine"**: contains the Unreal Engine project and the generated simulation executable.

- **"detection_utils.py"**: contains the "run_detection" function for running the object detection on the rendered images.

- **"light_utils.py"**: contains the "csv_to_png" function for converting CSV files to PNG images.

- **"unreal_utils.py"**: contains the "run_unreal_executable" function for running the Unreal Engine executable.

- **"main.py"**: the main script for running the pipeline process.




