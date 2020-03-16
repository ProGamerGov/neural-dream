# neural-dream Installation

This guide will walk you through multiple ways to setup `neural-dream` on Ubuntu and Windows. If you wish to install PyTorch and neural-dream on a different operating system like MacOS, installation guides can be found [here](https://pytorch.org).

Note that in order to reduce their size, the pre-packaged binary releases (pip, Conda, etc...) have removed support for some older GPUs, and thus you will have to install from source in order to use these GPUs.


# Ubuntu:

## With A Package Manager:

The pip and Conda packages ship with CUDA and cuDNN already built in, so after you have installed PyTorch with pip or Conda, you can skip to [installing neural-dream](https://github.com/ProGamerGov/neural-dream/blob/master/INSTALL.md#install-neural-dream).

### pip:

The neural-dream PyPI page can be found here: https://pypi.org/project/neural-dream/

If you wish to install neural-dream as a pip package, then use the following command:

```
# in a terminal, run the command
pip install neural-dream
```

Or:


```
# in a terminal, run the command
pip3 install neural-dream
```

Next download the models with:


```
neural-dream -download_models
```

By default the models are downloaded to your home directory, but you can specify a download location with:

```
neural-dream -download_models -download_path <download_path>
```

To download specific models or specific groups of models, you can use a comma separated list of models like this:

```
neural-dream -download_models all-caffe-googlenet,caffe-vgg19
```

To print all the models available for download, run the following command:

```
neural-dream -download_models print-all
```

#### Github and pip:

Following the pip installation instructions
[here](http://pytorch.org), you can install PyTorch with the following commands:

```
# in a terminal, run the commands
cd ~/
pip install torch torchvision
```

Or:

```
cd ~/
pip3 install torch torchvision
```

Now continue on to [installing neural-dream](https://github.com/ProGamerGov/neural-dream/blob/master/INSTALL.md#install-neural-dream) to install neural-dream.

### Conda:

Following the Conda installation instructions
[here](http://pytorch.org), you can install PyTorch with the following command:

```
conda install pytorch torchvision -c pytorch
```

Now continue on to [installing neural-dream](https://github.com/ProGamerGov/neural-dream/blob/master/INSTALL.md#install-neural-dream) to install neural-dream.

## From Source:

### (Optional) Step 1: Install CUDA

If you have a [CUDA-capable GPU from NVIDIA](https://developer.nvidia.com/cuda-gpus) then you can
speed up `neural-dream` with CUDA.

Instructions for downloading and installing the latest CUDA version on all supported operating systems, can be found [here](https://developer.nvidia.com/cuda-downloads).


### (Optional) Step 2: Install cuDNN

cuDNN is a library from NVIDIA that efficiently implements many of the operations (like convolutions and pooling)
that are commonly used in deep learning.

After registering as a developer with NVIDIA, you can [download cuDNN here](https://developer.nvidia.com/cudnn). Make sure that you use the appropriate version of cuDNN for your version of CUDA.

Follow the download instructions on Nvidia's site to install cuDNN correctly.

Note that the cuDNN backend can only be used for GPU mode.

### (Optional) Steps 1-3: Install PyTorch with support for AMD GPUs using Radeon Open Compute Stack (ROCm)


It is recommended that if you wish to use PyTorch with an AMD GPU, you install it via the official ROCm dockerfile:
https://rocm.github.io/pytorch.html

- Supported AMD GPUs for the dockerfile are: Vega10 / gfx900 generation discrete graphics cards (Vega56, Vega64, or MI25).

PyTorch does not officially provide support for compilation on the host with AMD GPUs, but [a user guide posted here](https://github.com/ROCmSoftwarePlatform/pytorch/issues/337#issuecomment-467220107) apparently works well.

ROCm utilizes a CUDA porting tool called HIP, which automatically converts CUDA code into HIP code. HIP code can run on both AMD and Nvidia GPUs.


### Step 3: Install PyTorch

To install PyTorch [from source](https://github.com/pytorch/pytorch#from-source) on Ubuntu (Instructions may be different if you are using a different OS):

```
cd ~/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

cd ~/
git clone --recursive https://github.com/pytorch/vision
cd vision
python setup.py install
```

To check that your torch installation is working, run the command `python` or `python3` to enter the Python interpreter. Then type `import torch` and hit enter.

You can then type `print(torch.version.cuda)` and `print(torch.backends.cudnn.version())` to confirm that you are using the desired versions of CUDA and cuDNN.

To quit just type `exit()` or use  Ctrl-D.

Now continue on to [installing neural-dream](https://github.com/ProGamerGov/neural-dream/blob/master/INSTALL.md#install-neural-dream) to install neural-dream.


# Windows Installation

If you wish to install PyTorch on Windows From Source or via Conda, you can find instructions on the PyTorch website: https://pytorch.org/


### Github and pip

First, you will need to download Python 3 and install it: https://www.python.org/downloads/windows/. I recommend using the executable installer for the latest version of Python 3.

Then using https://pytorch.org/, get the correct pip command, paste it into the Command Prompt (CMD) and hit enter:


```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```


After installing PyTorch, download the neural-dream Github repository and extract/unzip it to the desired location.

Then copy the file path to your neural-dream folder, and paste it into the Command Prompt, with `cd` in front of it and then hit enter.

In the example below, the neural-dream folder was placed on the desktop:

```
cd C:\Users\<User_Name>\Desktop\neural-dream-master
```

You can now continue on to [installing neural-dream](https://github.com/ProGamerGov/neural-dream/blob/master/INSTALL.md#install-neural-dream), skipping the `git clone` step.

# Install neural-dream

First we clone `neural-dream` from GitHub:

```
cd ~/
git clone https://github.com/ProGamerGov/neural-dream.git
cd neural-dream
```

Next we need to download the pretrained neural network models:

```
python models/download_models.py
```

You should now be able to run `neural-dream` in CPU mode like this:

```
python neural_dream.py -gpu c -print_octave_iter 2
```

If you installed PyTorch with support for CUDA, then should now be able to run `neural-dream` in GPU mode like this:

```
python neural_dream.py -gpu 0 -print_octave_iter 5
```

If you installed PyTorch with support for cuDNN, then you should now be able to run `neural-dream` with the `cudnn` backend like this:

```
python neural_dream.py -gpu 0 -backend cudnn -print_octave_iter 5
```

If everything is working properly you should see output like this:

```
Octave iter 1 iteration 25 / 50
  DeepDream 1 loss: 19534752.0
Octave iter 1 iteration 50 / 50
  DeepDream 1 loss: 23289720.0
Octave iter 2 iteration 25 / 50
  DeepDream 1 loss: 38870436.0
Octave iter 2 iteration 50 / 50
  DeepDream 1 loss: 47514664.0
Iteration 1 / 10
  DeepDream 1 loss: 71727704.0
  Total loss: 2767866014.0
Octave iter 1 iteration 25 / 50
  DeepDream 1 loss: 27209894.0
Octave iter 1 iteration 50 / 50
  DeepDream 1 loss: 31386542.0
Octave iter 2 iteration 25 / 50
  DeepDream 1 loss: 47773244.0
Octave iter 2 iteration 50 / 50
  DeepDream 1 loss: 51204812.0
Iteration 2 / 10
  DeepDream 1 loss: 87182300.0
  Total loss: 3758961954.0
```