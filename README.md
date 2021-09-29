# neural-dream
<a href="https://replicate.ai/progamergov/neural-dream"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=20></a>


This is a PyTorch implementation of DeepDream. The code is based on [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt).

<div align="center">
 <img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/places365_big.png" width="710px">
</div>

Here we DeepDream a photograph of the Golden Gate Bridge with a variety of settings:

<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/inputs/golden_gate.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/goldengate_3a_5x5_reduce.png" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/places205_4b_pool_proj.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/places365_inception_4a_pool_proj.png" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/goldengate_4d_5x5_s10.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/goldengate_4d_3x3_reduce_avg10_lp4.png" height="250px">
</div>


### Specific Channel Selection

You can select individual or specific combinations of channels.

Clockwise from upper left: 119, 1, 29, and all channels of the `inception_4d_3x3_reduce` layer

<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c119.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c1.png" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_4d_3x3_reduce_call.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c29.png" height="250px">

</div>

Clockwise from upper left: 25, 108, 25 & 108, and 25 & 119 from the `inception_4d_3x3_reduce` layer

<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c25.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c108.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c25_119.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_c25_108.png" height="250px">
</div>

### Channel Selection Based On Activation Strength

You can select channels automatically based on their activation strength.

Clockwise from upper left: The top 10 weakest channels, the 10 most average channels,
the top 10 strongest channels, and all channels of the `inception_4e_3x3_reduce` layer


<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_w10.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_avg10.png" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_4e_3x3_reduce_all.png" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/tubingen_s10.png" height="250px">

</div>


## Setup:

Dependencies:
* [PyTorch](http://pytorch.org/)

Optional dependencies:
* For CUDA backend:
  * CUDA 7.5 or above
* For cuDNN backend:
  * cuDNN v6 or above
* For ROCm backend:
  * ROCm 2.1 or above
* For MKL backend:
  * MKL 2019 or above
* For OpenMP backend:
  * OpenMP 5.0 or above

After installing the dependencies, you'll need to run the following script to download the BVLC GoogleNet model:
```
python models/download_models.py
```
This will download the original [BVLC GoogleNet model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).

If you have a smaller memory GPU then using the NIN Imagenet model could be an alternative to the BVLC GoogleNet model, though it's DeepDream quality is nowhere near that of the other models. You can get the details on the model from [BVLC Caffe ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). The NIN model is downloaded when you run the `download_models.py` script with default parameters.

To download most of the compatible models, run the `download_models.py` script with following parameters:

```
python models/download_models.py -models all
```

You can find detailed installation instructions for Ubuntu and Windows in the [installation guide](INSTALL.md).

## Usage
Basic usage:
```
python neural_dream.py -content_image <image.jpg>
```

cuDNN usage with NIN Model:
```
python neural_dream.py -content_image examples/inputs/brad_pitt.jpg -output_image pitt_nin_cudnn.png -model_file models/nin_imagenet.pth -gpu 0 -backend cudnn -num_iterations 10 -seed 876 -dream_layers relu0,relu3,relu7,relu12 -dream_weight 10 -image_size 512 -optimizer adam -learning_rate 0.1
```

![cuDNN NIN Model Picasso Brad Pitt](https://raw.githubusercontent.com/ProGamerGov/neural-dream/master/examples/outputs/pitt_nin_cudnn.png)


Note that paths to images should not contain the `~` character to represent your home directory; you should instead use a relative
path or a full absolute path.

**Options**:
* `-image_size`: Maximum side length (in pixels) of the generated image. Default is 512.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to `c`.

**Optimization options**:
* `-dream_weight`: How much to weight DeepDream. Default is `1e3`.
* `-tv_weight`: Weight of total-variation (TV) regularization; this helps to smooth the image.
  Default is set to `0` to disable total-variation (TV) regularization.
* `-l2_weight`: Weight of latent state regularization.
  Default is set to `0` to disable latent state regularization.
* `-num_iterations`: Default is `10`.
* `-init`: Method for generating the generated image; one of `random` or `image`.
  Default is `image` which initializes with the content image; `random` uses random noise to initialize the input image.
* `-jitter`: Apply jitter to image. Default is `32`. Set to `0` to disable jitter.
* `-layer_sigma`: Apply gaussian blur to image. Default is set to `0` to disable the gaussian blur layer.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `adam`.
  Adam tends to perform the best for DeepDream. L-BFGS tends to give worse results and it uses more memory; when using L-BFGS you will probably need to play with other parameters to get good results, especially the learning rate.
* `-learning_rate`: Learning rate to use with the ADAM and L-BFGS optimizers. Default is `1.5`. On other DeepDream projects this parameter is commonly called 'step size'.
* `-normalize_weights`: If this flag is present, dream weights will be divided by the number of channels for each layer. Idea from [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer).
* `-loss_mode`: The DeepDream loss mode; `bce`, `mse`, `mean`, `norm`, or `l2`; default is `l2`.

**Output options**:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-output_start_num`: The number to start output image names at. Default is set to `1`.
* `-leading_zeros`: The number of leading zeros to use for output image names. Default is set to `0` to disable leading zeros.
* `-print_iter`: Print progress every `print_iter` iterations. Set to `0` to disable printing.
* `-print_octave_iter`: Print octave progress every `print_octave_iter` iterations. Default is set to `0` to disable printing. If tiling is enabled, then octave progress will be printed every `print_octave_iter` octaves.
* `-save_iter`: Save the image every `save_iter` iterations. Set to `0` to disable saving intermediate results.
* `-save_octave_iter`: Save the image every `save_octave_iter` iterations. Default is set to `0` to disable saving intermediate results. If tiling is enabled, then octaves will be saved every `save_octave_iter` octaves.

**Layer options**:
* `-dream_layers`: Comma-separated list of layer names to use for DeepDream reconstruction.

**Channel options:**
* `-channels`: Comma-separated list of channels to use for DeepDream. If `-channel_mode` is set to a value other than `all` or `ignore`, only the first value in the list will be used.
* `-channel_mode`: The DeepDream channel selection mode; `all`, `strong`, `avg`, `weak`, or `ignore`; default is `all`. The `strong` option will select the strongest channels, while `weak` will do the same with the weakest channels. The `avg` option will select the most average channels instead of the strongest or weakest. The number of channels selected by `strong`, `avg`, or `weak` is based on the first value for the `-channels` parameter. The `ignore` option will omit any specified channels.
* `-channel_capture`: How often to select channels based on activation strength; either `once` or `octave_iter`; default is `once`. The `once` option will select channels once at the start, while the `octave_iter` will select potentially new channels every octave iteration. This parameter only comes into play if `-channel_mode` is not set to `all` or `ignore`.

**Octave options:**
* `-num_octaves`: Number of octaves per iteration. Default is `4`.
* `-octave_scale`: Value for resizing the image by. Default is `0.6`.
* `-octave_iter`: Number of iterations per octave. Default is `50`. On other DeepDream projects this parameter is commonly called 'steps'.
* `-octave_mode`: The octave size calculation mode; `normal`, `advanced`, `manual_max`, `manual_min`, or `manual`. Default is `normal`. If set to `manual_max` or `manual_min`, then `-octave_scale` takes a comma separated list of image sizes for the largest or smallest image dimension for `num_octaves` minus 1 octaves. If set `manual` then `-octave_scale` takes a comma separated list of image size pairs for  `num_octaves` minus 1 octaves, in the form of `<Height>,<Width>`.

**Laplacian Pyramid options:**
* `-lap_scale`: The number of layers in a layer's laplacian pyramid. Default is set to `0` to disable laplacian pyramids.
* `-sigma`: The strength of gaussian blur to use in laplacian pyramids. Default is `1`. By default, unless a second sigma value is provided with a comma to separate it from the first, the high gaussian layers will use sigma `sigma` * `lap_scale`.

**Zoom options:**
* `-zoom`: The amount to zoom in on the image.
* `-zoom_mode`: Whether to read the zoom value as a percentage or pixel value; one of `percentage` or `pixel`. Default is `percentage`.

**FFT options:**
* `-use_fft`: Whether to enable Fast Fourier transform (FFT) decorrelation.
* `-fft_block`: The size of your FFT frequency filtering block. Default is `25`.

**Tiling options:**
* `-tile_size`: The desired tile size to use. Default is set to `0` to disable tiling.
* `-overlap_percent`: The percentage of overlap to use for the tiles. Default is `50`.
* `-print_tile`: Print the current tile being processed every `print_tile` tiles without any other information. Default is set to `0` to disable printing.
* `-print_tile_iter`: Print tile progress every `print_tile_iter` iterations. Default is set to `0` to disable printing.
* `-image_capture_size`: The image size to use for the initial full image capture and optional `-classify` parameter. Default is set to `512`. Set to `0` disable it and `image_size` is used instead.

**GIF options:**
* `-create_gif`: Whether to create a GIF from the output images after all iterations have been completed.
* `-frame_duration`: The duration for each GIF frame in milliseconds. Default is `100`.

**Help options:**
* `-print_layers`: Pass this flag to print the names of all usable layers for the selected model.
* `-print_channels`: Pass this flag to print all the selected channels.

**Other options**:
* `-original_colors`: If you set this to `1`, then the output image will keep the colors of the content image.
* `-model_file`: Path to the `.pth` file for the VGG Caffe model. Default is the original VGG-19 model; you can also try the original VGG-16 model.
* `-model_type`: Whether the model was trained using Caffe, PyTorch, or Keras preprocessing; `caffe`, `pytorch`, `keras`, or `auto`; default is `auto`.
* `-model_mean`: A comma separated list of 3 numbers for the model's mean; default is `auto`.
* `-pooling`: The type of pooling layers to use for VGG and NIN models; one of `max` or `avg`. Default is `max`. VGG models seem to create better results with average pooling.
* `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.
* `-multidevice_strategy`: A comma-separated list of layer indices at which to split the network when using multiple devices. See [Multi-GPU scaling](https://github.com/ProGamerGov/neural-dream#multi-gpu-scaling) for more details. Currently this feature only works for VGG and NIN models.
* `-backend`: `nn`, `cudnn`, `openmp`, or `mkl`. Default is `nn`. `mkl` requires Intel's MKL backend.
* `-cudnn_autotune`: When using the cuDNN backend, pass this flag to use the built-in cuDNN autotuner to select
  the best convolution algorithms for your architecture. This will make the first iteration a bit slower and can
  take a bit more memory, but may significantly speed up the cuDNN backend.
* `-clamp`: If this flag is enabled, every iteration will clamp the output image so that it is within the model's input range.
* `-adjust_contrast`: A value between `0` and `100.0` for altering the image's contrast (ex: `99.98`). Default is set to 0 to disable contrast adjustments.
* `-label_file`:  Path to the `.txt` category list file for classification and channel selection.
* `-random_transforms`: Whether to use random transforms on the image; either `none`, `rotate`, `flip`, or `all`; default is `none`.
* `-classify`: Display what the model thinks an image contains. Integer for the number of choices ranked by how likely each is.


## Frequently Asked Questions

**Problem:** The program runs out of memory and dies

**Solution:** Try reducing the image size: `-image_size 512` (or lower). Note that different image sizes will likely
require non-default values for `-octave_scale` and `-num_octaves` for optimal results.
If you are running on a GPU, you can also try running with `-backend cudnn` to reduce memory usage.

**Problem:** `-backend cudnn` is slower than default NN backend

**Solution:** Add the flag `-cudnn_autotune`; this will use the built-in cuDNN autotuner to select the best convolution algorithms.

**Problem:** Get the following error message:

`Missing key(s) in state_dict: "classifier.0.bias", "classifier.0.weight", "classifier.3.bias", "classifier.3.weight".
        Unexpected key(s) in state_dict: "classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias".`

**Solution:** Due to a mix up with layer locations, older models require a fix to be compatible with newer versions of PyTorch. The included [`donwload_models.py`](https://github.com/ProGamerGov/neural-dream/blob/master/models/download_models.py) script will automatically perform these fixes after downloading the models.

**Problem:** Get the following error message:

`Given input size: (...). Calculated output size: (...). Output size is too small`

**Solution:** Use a larger `-image_size` value and/or adjust the octave parameters so that the smallest octave size is larger.

## Memory Usage
By default, `neural-dream` uses the `nn` backend for convolutions and Adam for optimization. These give good results, but can both use a lot of memory. You can reduce memory usage with the following:

* **Use cuDNN**: Add the flag `-backend cudnn` to use the cuDNN backend. This will only work in GPU mode.
* **Reduce image size**: You can reduce the size of the generated image to lower memory usage;
  pass the flag `-image_size 256` to generate an image at half the default size.

With the default settings, neural-dream uses about 1.3 GB of GPU memory on my system; switching to cuDNN reduces the GPU memory footprint to about 1 GB.


## Multi-GPU scaling
You can use multiple CPU and GPU devices to process images at higher resolutions; different layers of the network will be
computed on different devices. You can control which GPU and CPU devices are used with the `-gpu` flag, and you can control
how to split layers across devices using the `-multidevice_strategy` flag.

For example in a server with four GPUs, you can give the flag `-gpu 0,1,2,3` to process on GPUs 0, 1, 2, and 3 in that order; by also giving the flag `-multidevice_strategy 3,6,12` you indicate that the first two layers should be computed on GPU 0, layers 3 to 5 should be computed on GPU 1, layers 6 to 11 should be computed on GPU 2, and the remaining layers should be computed on GPU 3. You will need to tune the `-multidevice_strategy` for your setup in order to achieve maximal resolution.

We can achieve very high quality results at high resolution by combining multi-GPU processing with multiscale
generation as described in the paper
<a href="https://arxiv.org/abs/1611.07865">**Controlling Perceptual Factors in Neural Style Transfer**</a> by Leon A. Gatys,
Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann and Eli Shechtman.
