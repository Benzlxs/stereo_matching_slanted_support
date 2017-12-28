# Stereo_matching_slanted_support
This is the code in paper [Efficient Methods Using Slanted Windows for Slanted Surfaces](http://apps.webofknowledge.com/Search.do?product=WOS&SID=C5oqWs1CGyfqY8n42RB&search_mode=GeneralSearch&prID=e0079d24-9970-4978-b331-45a7e5e80791
), which I pulished when I did research on stereo matching during master study.

## Brief introduction of algorithm
The cost computation: absolute difference (AD)  +  census transform (CT).

The cost aggregation: bilateral filter is used to adapte weight according the slanted support window in disparity space.

The post processing: outlier detection, outlier filling and refinement of disparities.


## Compliation
The code is a Visual Studio 2010 project on Windows x64 platform. To build the project, you need to configure OpenCV. (>=version 2.4.6). The code requires no platform-dependent libraries. Thus, it is easy to compile it on other platforms with OpenCV.

## Citation
Citation is very important for researchers. If you find this code useful, please cite:
```
@inproceedings{stereo_matching_slanted_support,
        author    = {Xuesong LI and Jianguo Liu and Guang Chen and Heng Fu},
        title     = {Efficient Methods Using Slanted Support Windows for Slanted Surfaces},
        journal   = {IET Computer Vision},
        year      = 2016,
        pages     = {384-391},
        month     = 8,
        note      = {http://apps.webofknowledge.com/Search.do?product=WOS&SID=C5oqWs1CGyfqY8n42RB&search_mode=GeneralSearch&prID=e0079d24-9970-4978-b331-45a7e5e80791}, 
        volume    = 10
}
```
