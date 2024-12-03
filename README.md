# Deepstream-Ultralytics

In this guide, we’ll explore how to run a trained model on NVIDIA devices using DeepStream for real-time inference. We’ll walk through using Roboflow to prepare the dataset, train the model with Ultralytics, and deploy it on an NVIDIA device using DeepStream. We’ll also cover installing NVIDIA JetPack SDK and setting up essential tools like CUDA, TensorRT, PyTorch, torchvision, and jtop.

# Step 1: Preparing the Dataset with Roboflow

Sign up and Create a Project
Visit Roboflow and create a new project. Upload your dataset, label the images, and choose an export format compatible with Ultralytics YOLO.
Generate Dataset
Apply preprocessing steps such as resizing, augmentations, and annotation conversion. Download the dataset in YOLO format.

# Step 2: Training the Model with Ultralytics YOLO

Install Ultralytics
Install the Ultralytics package using Python:
```
pip install ultralytics
```

2. Train the Model
Use the downloaded dataset from Roboflow:
```
yolo task=detect mode=train data=/path/to/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```
Once training is complete, save the best model weights (best.pt) for deployment.

# Step 3: Setting Up the NVIDIA Device

Deepstream — Workflow
1. Install NVIDIA JetPack SDK
JetPack provides essential tools like CUDA and TensorRT. To install:

Download the NVIDIA SDK Manager from NVIDIA Developer.
Connect your NVIDIA Jetson device and flash the SD card with the latest JetPack version. (Used Version = Jetpack 5.1.2)
Reboot the device after installation.

2. Install CUDA and TensorRT
CUDA and TensorRT are installed with JetPack, but verify the installation:
```
nvcc --version    # To check CUDA version
dpkg -l | grep nvinfer # To check Tensorrt
```
If not, You can manually install the CUDA and TensorRT, By following the below steps

CUDA Toolkit 11.4 Downloads
Select Linux or Windows operating system and download CUDA Toolkit 11.4.
developer.nvidia.com

# CUDA = 11.04
```
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux_sbsa.run
sudo sh cuda_11.4.0_470.42.01_linux_sbsa.run
#TensorRT
sudo apt update
sudo apt install nvidia-tensorrt
```

3. Install PyTorch and torchvision
Install compatible versions of PyTorch and torchvision for Jetson devices:

we need to manually install the pre-built PyTorch pip wheel and compile/ install Torchvision from the source.

Visit this page to access all the PyTorch and Torchvision links.

Here are some of the versions supported by JetPack 5.0 and above.

PyTorch v1.11.0
Supported by JetPack 5.0 (L4T R34.1.0) / JetPack 5.0.1 (L4T R34.1.1) / JetPack 5.0.2 (L4T R35.1.0) with Python 3.8

file_name: torch-1.11.0-cp38-cp38-linux_aarch64.whl URL:

https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl

PyTorch v1.12.0
Supported by JetPack 5.0 (L4T R34.1.0) / JetPack 5.0.1 (L4T R34.1.1) / JetPack 5.0.2 (L4T R35.1.0) with Python 3.8

file_name: torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl URL:

https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

Step 1. Install torch according to your JetPack version in the following format
```
wget <URL> -O <file_name>
pip3 install <file_name>
```
For example, here we are running JP5.0.2 and therefore we choose PyTorch v1.12.0
```
sudo apt-get install -y libopenblas-base libopenmpi-dev
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl -O torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
```
Step 2. Install torchvision depending on the version of PyTorch that you have installed. For example, we chose PyTorch v1.12.0, which means, we need to choose Torchvision v0.13.0
```
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install --user
```
Here is a list of the corresponding torchvision versions that you need to install according to the PyTorch version:

PyTorch v1.11 — torchvision v0.12.0
PyTorch v1.12 — torchvision v0.13.0

4. Install jtop
Monitor your Jetson device resources using jtop:
```
sudo apt update
sudo apt install python3-pip -y
pip3 install jetson-stats
jtop
```
# Step 4: Setting Up DeepStream
Install DeepStream​
There are multiple ways of installing DeepStream to the Jetson device. You can follow this guide to learn more. However, we recommend you to install DeepStream via the SDK Manager because it can guarantee for a successful and easy installation.

https://docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Quickstart.html

DeepStream | NVIDIA NGC
DeepStream SDK delivers a complete streaming analytics toolkit for AI based video and image understanding and…
catalog.ngc.nvidia.com
```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/6.3/files?redirect=true&path=deepstream-6.3_6.3.0-1_arm64.deb' -O deepstream-6.3_6.3.0-1_arm64.deb
sudo apt-get install ./deepstream-6.3_6.3.0-1_arm64.deb
deepstream-app version
```
If you install DeepStream using SDK manager, you need to execute the below commands which are additional dependencies for DeepStream, after the system boots up
```
sudo apt install \
libssl1.1 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev
```
Install Necessary Packages​
Step 1. Access the terminal of the Jetson device, install pip, and upgrade it
```
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
```
Step 2. Clone the following repo
git clone https://github.com/ultralytics/ultralytics.git
DeepStream Configuration for YOLOv8​
Step 1. Clone the following repo
```
cd ~
git clone https://github.com/marcoslucianops/DeepStream-Yolo
```
Step 2. Checkout the repo to the following commit
```
cd DeepStream-Yolo
git checkout 68f762d5bdeae7ac3458529bfe6fed72714336ca
```
Step 3. Copy gen_wts_yoloV8.py from DeepStream-Yolo/utils into ultralytics directory
```
cp utils/gen_wts_yoloV8.py ~/ultralytics
```
Step 4. Inside the ultralytics repo, download pt file from YOLOv8 releases (example for YOLOv8s)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
NOTE: You can use your custom model, but it is important to keep the YOLO model reference (yolov8_) in your cfg and weights/wts filenames to generate the engine correctly.

Step 5. Generate the cfg, wts and labels.txt (if available) files (example for YOLOv8s)
```
python3 gen_wts_yoloV8.py -w yolov8s.pt
```
Note: To change the inference size (defaut: 640)

-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
Example for 1280:
-s 1280
or
-s 1280 1280

Step 6. Copy the generated cfg, wts and labels.txt (if generated) files into the DeepStream-Yolo folder
```
cp yolov8s.cfg ~/DeepStream-Yolo
cp yolov8s.wts ~/DeepStream-Yolo
cp labels.txt ~/DeepStream-Yolo
```
Step 7. Open the DeepStream-Yolo folder and compile the library
```
cd ~/DeepStream-Yolo
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo  # for DeepStream 6.2/ 6.1.1 / 6.1
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo  # for DeepStream 6.0.1 / 6.0
```
Step 8. Edit the config_infer_primary_yoloV8.txt file according to your model (example for YOLOv8s with 80 classes)
[property]
...
custom-network-config=yolov8s.cfg
model-file=yolov8s.wts
...
num-detected-classes=80
...
Step 9. Edit the deepstream_app_config.txt file
...
[primary-gie]
...
config-file=config_infer_primary_yoloV8.txt
Step 10. Change the video source in deepstream_app_config.txt file. Here a default video file is loaded as you can see below
...
[source0]
...
uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

# Step 5: Run the Inference​
```
deepstream-app -c deepstream_app_config.txt
```
Launch DeepStream to perform real-time inference with your trained YOLO model. Monitor the performance using jtop to ensure efficient utilization of your NVIDIA device.
