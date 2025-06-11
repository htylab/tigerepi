## Background

* This repo provides deep-learning methods for EPI VDM corrention and EPI brain segmentaion.
* We also provided the stand-alone application working on Windows, Mac, and Linux.

### Install stand-alone version
https://github.com/htylab/tigerepi/releases

### Usage

    tigerepi -bmawk c:\data\*.nii.gz -o c:\output

### As a python package

    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerepi/archive/refs/heads/main.zip

## Segmentation

### As a python package

    import tigerepi
    tigerepi.seg('bmawk', r'C:\EPI_dir', r'C:\output_dir')
    tigerepi.seg('bmawk', r'C:\EPI_dir\**\*.nii.gz', r'C:\output_dir')
    tigerepi.seg('bmawk', r'C:\EPI_dir\**\*.nii.gz') # storing output in the same dir
    tigerepi.seg('ag', r'C:\EPI_dir') # Producing aseg masks with GPU


** Mac and Windows  are supported.**

** Ubuntu (version >20.04)  are supported.**

```
>>tigerepi  c:\data\**\*epi.nii -o c:\outputdir -b -m -a -w -k
-b: producing extracted brain
-m: producing the brain mask
-a: producing the aseg mask
-w, Producing the white matter parcellation (work in progress)
-k, Producing the dkt mask (work in progress)
```

## Virtual Displacement Mapping

### As a python package

    import tigerepi
    tigerepi.vdm(r'C:\EPI_dir', r'C:\output_dir', b0_index=0)

** Mac and Windows  are supported.**

** Ubuntu (version >20.04)  are supported.**
```
>>tigerepi_vdm  c:\data\**\*epi.nii -o c:\outputdir
```
- For additional options type:
```
>>tigerepi_vdm -h
```

## Citation

* If you use this application, cite the following paper:

1. Kuo CC, Huang TY, Lin YR, Chuang TC, Tsai SY, Chung HW, “Referenceless correction of EPI distortion with virtual displacement mapping”

## Label definitions

For label definitions, please check here. [Label definitions](doc/seglabel.md)
