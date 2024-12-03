<<<<<<< HEAD
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
=======
# DeepStream-Yolo

NVIDIA DeepStream SDK 6.1 / 6.0.1 / 6.0 configuration for YOLO models

### Future updates

* Models benchmarks
* DeepStream tutorials
* YOLOX support
* YOLOv6 support
* YOLOv7 support
* Dynamic batch-size

### Improvements on this repository

* Darknet cfg params parser (no need to edit `nvdsparsebbox_Yolo.cpp` or other files)
* Support for `new_coords` and `scale_x_y` params
* Support for new models
* Support for new layers
* Support for new activations
* Support for convolutional groups
* Support for INT8 calibration
* Support for non square models
* New documentation for multiple models
* YOLOv5 support
* YOLOR support
* **GPU YOLO Decoder** [#138](https://github.com/marcoslucianops/DeepStream-Yolo/issues/138)
* **GPU Batched NMS** [#142](https://github.com/marcoslucianops/DeepStream-Yolo/issues/142)
* **PP-YOLOE support**

##

### Getting started

* [Requirements](#requirements)
* [Suported models](#supported-models)
* [Benchmarks](#benchmarks)
* [dGPU installation](#dgpu-installation)
* [Basic usage](#basic-usage)
* [NMS configuration](#nms-configuration)
* [INT8 calibration](#int8-calibration)
* [YOLOv5 usage](docs/YOLOv5.md)
* [YOLOR usage](docs/YOLOR.md)
* [PP-YOLOE usage](docs/PPYOLOE.md)
* [Using your custom model](docs/customModels.md)
* [Multiple YOLO GIEs](docs/multipleGIEs.md)

##

### Requirements

#### DeepStream 6.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.6 Update 1](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.2 GA Update 4 (8.2.5.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 510.47.03](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/deepstream-getting-started)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4 Update 1](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver >= 470.63.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on Jetson platform

* [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on Jetson platform

* [JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

##

### Suported models

* [Darknet YOLO](https://github.com/AlexeyAB/darknet)
* [YOLOv5 >= 2.0](https://github.com/ultralytics/yolov5)
* [YOLOR](https://github.com/WongKinYiu/yolor)
* [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
* [MobileNet-YOLO](https://github.com/dog-qiuqiu/MobileNet-Yolo)
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)

##

### Benchmarks

New tests comming soon.

##

### dGPU installation

To install the DeepStream on dGPU (x86 platform), without docker, we need to do some steps to prepare the computer.

<details><summary>DeepStream 6.1</summary>

#### 1. Disable Secure Boot in BIOS

#### 2. Install dependencies

```
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc (replace $CUDA_PATH to your CUDA path)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Install CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

#### 4. Download and install NVIDIA Driver

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
  ```

* Data center / Tesla series

  ```
  wget https://us.download.nvidia.com/tesla/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
  ```

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 5. Download and install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run
sudo sh cuda_11.6.1_510.47.03_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

#### 6. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT

TensorRT 8.2 GA Update 4 for Ubuntu 20.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4 and 11.5 DEB local repo Package

```
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505/82307095.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 libnvparsers8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 libnvinfer-bin=8.2.5-1+cuda11.4 libnvinfer-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 libnvparsers-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 libnvinfer-samples=8.2.5-1+cuda11.4 libnvinfer-doc=8.2.5-1+cuda11.4 libcudnn8-dev=8.4.0.27-1+cuda11.6 libcudnn8=8.4.0.27-1+cuda11.6
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-getting-started) and install the DeepStream SDK

DeepStream 6.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.1_6.1.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.6 /usr/local/cuda
```

#### 8. Reboot the computer

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.0.1 / 6.0</summary>

#### 1. Disable Secure Boot in BIOS

<details><summary>If you are using a laptop with newer Intel/AMD processors and your Graphics in Settings->Details->About tab is llvmpipe, please update the kernel.</summary>

```
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100_5.11.0-051100.202102142330_all.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-image-unsigned-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-modules-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
sudo dpkg -i  *.deb
sudo reboot
```

</details>

#### 2. Install dependencies

```
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Install DKMS only if you are using the default Ubuntu kernel

```
sudo apt-get install dkms
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc (replace $CUDA_PATH to your CUDA path)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Install CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

#### 4. Download and install NVIDIA Driver

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
  ```

* Data center / Tesla series

  ```
  wget https://us.download.nvidia.com/tesla/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
  ```

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 5. Download and install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
sudo sh cuda_11.4.1_470.57.02_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

#### 6. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT

TensorRT 8.0.1 GA for Ubuntu 18.04 and CUDA 11.3 DEB local repo package

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626/7fa2af80.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.0.1-1+cuda11.3 libnvinfer-plugin8=8.0.1-1+cuda11.3 libnvparsers8=8.0.1-1+cuda11.3 libnvonnxparsers8=8.0.1-1+cuda11.3 libnvinfer-bin=8.0.1-1+cuda11.3 libnvinfer-dev=8.0.1-1+cuda11.3 libnvinfer-plugin-dev=8.0.1-1+cuda11.3 libnvparsers-dev=8.0.1-1+cuda11.3 libnvonnxparsers-dev=8.0.1-1+cuda11.3 libnvinfer-samples=8.0.1-1+cuda11.3 libnvinfer-doc=8.0.1-1+cuda11.3 libcudnn8-dev=8.2.1.32-1+cuda11.3 libcudnn8=8.2.1.32-1+cuda11.3
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived) and install the DeepStream SDK

* DeepStream 6.0.1 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.1-1_amd64.deb
  ```

* DeepStream 6.0 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.0-1_amd64.deb
  ```

* Run

  ```
  rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
  sudo ln -snf /usr/local/cuda-11.4 /usr/local/cuda
  ```

#### 8. Reboot the computer

```
sudo reboot
```

</details>

##

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
```

#### 2. Download the `cfg` and `weights` files from [Darknet](https://github.com/AlexeyAB/darknet) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

#### 4. Edit the `config_infer_primary.txt` file according to your model (example for YOLOv4)

```
[property]
...
custom-network-config=yolov4.cfg
model-file=yolov4.weights
...
```

#### 5. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: If you want to use YOLOv2 or YOLOv2-Tiny models, change the `deepstream_app_config.txt` file before run it

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV2.txt
...
```

##

### NMS Configuration

To change the `iou-threshold`, `score-threshold` and `topk` values, modify the `config_nms.txt` file and regenerate the model engine file.

```
[property]
iou-threshold=0.45
score-threshold=0.25
topk=300
```

**NOTE**: Lower `topk` values will result in more performance.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

**NOTE**: You are still able to change the `pre-cluster-threshold` values in the config_infer files.

##

### INT8 calibration

#### 1. Install OpenCV

```
sudo apt-get install libopencv-dev
```

#### 2. Compile/recompile the `nvdsinfer_custom_impl_Yolo` lib with OpenCV support

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

#### 3. For COCO dataset, download the [val2017](https://drive.google.com/file/d/1gbvfn7mcsGDRZ_luJwtITL-ru2kK99aK/view?usp=sharing), extract, and move to DeepStream-Yolo folder

* Select 1000 random images from COCO dataset to run calibration

  ```
  mkdir calibration
  ```

  ```
  for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
      cp ${jpg} calibration/; \
  done
  ```

* Create the `calibration.txt` file with all selected images

  ```
  realpath calibration/*jpg > calibration.txt
  ```

* Set environment variables

  ```
  export INT8_CALIB_IMG_PATH=calibration.txt
  export INT8_CALIB_BATCH_SIZE=1
  ```

* Edit the `config_infer` file

  ```
  ...
  model-engine-file=model_b1_gpu0_fp32.engine
  #int8-calib-file=calib.table
  ...
  network-mode=0
  ...
  ```

    To

  ```
  ...
  model-engine-file=model_b1_gpu0_int8.engine
  int8-calib-file=calib.table
  ...
  network-mode=1
  ...
  ```

* Run

  ```
  deepstream-app -c deepstream_app_config.txt
  ```

**NOTE**: NVIDIA recommends at least 500 images to get a good accuracy. On this example, I used 1000 images to get better accuracy (more images = more accuracy). Higher `INT8_CALIB_BATCH_SIZE` values will result in more accuracy and faster calibration speed. Set it according to you GPU memory. This process can take a long time.

##

### Extract metadata

You can get metadata from DeepStream using Python and C/C++. For C/C++, you can edit the `deepstream-app` or `deepstream-test` codes. For Python, your can install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

Basically, you need manipulate the `NvDsObjectMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsObjectMeta.html)) `and NvDsFrameMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsFrameMeta.html)) to get the label, position, etc. of bboxes.

##

My projects: https://www.youtube.com/MarcosLucianoTV
>>>>>>> Add LICENSE
