import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural-dream",
    version="0.0.3",
    author="ProGamerGov",
    description="A PyTorch implementation of DeepDream",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='neural artistic neural-dream dream pytorch caffe neuralart neural-art hallucinations visualization feature-visualization mlart machine-learning-art aiart ai-art deepdream neuraldream pytorch-deepdream deepdream-pytorch tiling googlenet inception resnet vgg vgg16 vgg19 nin tensorflow keras inceptionv3 inception-v3',
    entry_points={
        'console_scripts': ["neural-dream = neural_dream.neural_dream:main"],
    },
    url="https://github.com/ProGamerGov/neural-dream/tree/pip-master/",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchvision', 'pillow'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Artistic Software",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
