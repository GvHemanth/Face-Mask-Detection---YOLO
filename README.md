# Face Mask Detection using YOLO

![download1](https://github.com/GvHemanth/Face-Mask-Detection---YOLO/assets/125199925/31c316df-df29-4c8c-b9d8-562be0c8e7d6)


## Overview

This project is aimed at building a custom object detector using the YOLO (You Only Look Once) algorithm for precise face mask detection in images. The system can help in ensuring compliance with face mask guidelines during the COVID-19 pandemic.

Key Features:
- Utilizes YOLO for real-time object detection.
- Customizes YOLO for face mask detection.
- Training on a custom dataset of images with annotations.
- High accuracy and efficiency in detecting face masks.

## Dependencies

To run this project, you'll need the following dependencies:

- Python 3
- OpenCV (cv2)
- Darknet (YOLO)
- Pandas
- Scikit-learn (for data splitting)
- CUDA and cuDNN (for GPU support)
- Google Colab (for cloud-based training, if desired)

## Dataset
You can find the dataset from this Link : https://drive.google.com/file/d/1q6GxNqZKNjyjkplHGNpWNZoHifFqHiMZ/view?usp=drive_link

## Getting Started

1. Clone the Darknet repository:

   ```
   git clone https://github.com/AlexeyAB/darknet.git
   cd darknet
   ```

2. Configure Darknet for GPU, CUDA, and OpenCV by modifying the Makefile:

   ```
   sed -i 's/GPU=0/GPU=1/g' Makefile
   sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
   sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
   ```

3. Create a directory for model storage in your Google Drive:

   ```
   mkdir ../drive/MyDrive/my_model/
   ```

4. Save a copy of the modified Makefile in the model directory:

   ```
   cp Makefile ../drive/MyDrive/my_model/
   ```

5. Mount your Google Drive:

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```

6. Create a directory for customization:

   ```
   mkdir customization
   cd customization
   ```

7. Fetch the pre-trained weights and config files:

   ```
   wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
   cp yolov4.weights ../../drive/MyDrive/my_model/
   cp ../cfg/yolov4.cfg .
   ```

8. Prepare your custom dataset by placing images and labels in the `custom_data/images/` and `custom_data/labels/` directories, respectively.

9. Configure class names and data file:

   ```
   touch training_data.txt
   touch validation_data.txt
   touch face_mask_classes.names
   touch face_mask.data
   echo "no face mask" >> face_mask_classes.names
   echo "face mask" >> face_mask_classes.names
   echo "classes = 2" >> face_mask.data
   echo "train = custom_data/training_data.txt" >> face_mask.data
   echo "valid = custom_data/validation_data.txt" >> face_mask.data
   echo "names = custom_data/face_mask_classes.names" >> face_mask.data
   echo "backup = backup/" >> face_mask.data
   cp face_mask.data ../../drive/MyDrive/my_model/
   cp face_mask_classes.names ../../drive/MyDrive/my_model/
   ```

10. Split your dataset into training and validation sets:

    ```
    python split_data.py
    ```

11. Compile the DarkNet project:

    ```
    cd ..
    make -j4
    ```

## Training the Model

1. Train the model using the following command:

   ```
   ./darknet detector train custom_data/face_mask.data customization/yolov4.cfg customization/yolov4.conv.137 -map -dont_show
   ```

2. To continue training from the last checkpoint:

   ```
   ./darknet detector train custom_data/face_mask.data customization/yolov4.cfg /content/drive/MyDrive/my_model/backup/yolov4_last.weights -map -dont_show
   ```

## Testing the Model

To test the model on an image, run:

```
./darknet detector test cfg/coco.data cfg/yolov4.cfg customization/yolov4.weights data/person.jpg
```

## Contributing

Contributions are welcome! Please open an issue or create a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

