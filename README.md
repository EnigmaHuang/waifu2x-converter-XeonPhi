# waifu2x on Xeon Phi(converter only version)

This is a reimplementation of waifu2x ([original](https://github.com/nagadomi/waifu2x)) converter function, in C & C++, using Intel's Xeon Phi accelerator.

An pure CPU version of this program is this(https://github.com/EnigmaHuang/waifu2x-converter-cpp). OpenCV is used for reading and writing image files but not for computing.

This project is inspired by this article(https://zhuanlan.zhihu.com/p/20390706), because @sakamoto-poteko(https://www.zhihu.com/people/poteko, GitHub :  https://github.com/sakamoto-poteko) decided not to write computer program any more. Wish him have a happy life.

NOTICE : This is a PROTOTYPE and may NOT have GOOD PERFORMANCE. 

## Dependencies

### Platform

 * Ubuntu / CentOS / Other x86_84 Linux with Intel CPU
 * Windows?
 
(Xeon Phi supports Windows but I have not tried)

### Libraries

 * [OpenCV](http://opencv.org/)(C++, version 3.1.0)

This programs also depends on libraries shown below, but these are already included in this repository.

 * [picojson](https://github.com/kazuho/picojson)
 * [TCLAP(Templatized C++ Command Line Parser Library)](http://tclap.sourceforge.net/)

## How to build

### for Linux

First you need to compile OpenCV 3.1.0, I used the commands below to do this:

`wget https://github.com/Itseez/opencv/archive/3.1.0.zip`

`unzip 3.1.0.zip`

`cd opencv-3.1.0`

`mkdir build`

`cd build`

`cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/enigma/opencv-3.1.0-gcc ..`

`make -j`

`make install`

I did not use pkg-config. Do not use ICC to compile OpenCV, beacuse OpenCV need to compile IPP and it may conflict with IPP in ICC. 

To compile this program, you need to have a Intel compiler, I use ICC 2015. Just type `make` in src/ directory. All the compiling flags have been set.

## Usage

Usage of this program can be seen by executing this with `--help` option.
