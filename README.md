## Background

* This repo provides deep-learning methods for EPI VDM corrention and brain segmentaion.
* We also provided the stand-alone application working on Windows, Mac, and Linux.

### Install stand-alone version
https://github.com/htylab/tigerepi/releases

### Usage

    tigerbx -bmad c:\data\*.nii.gz -o c:\output

### As a python package

    pip install onnxruntime #for gpu version: onnxruntime-gpu
    pip install https://github.com/htylab/tigerepi/archive/release.zip

### As a python package

## Segmentation

    import tigerepi
    tigerepi.run('bmaw', r'C:\T1w_dir', r'C:\output_dir')
    tigerepi.run('bmaw', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')
    tigerepi.run('bmaw', r'C:\T1w_dir\**\*.nii.gz') # storing output in the same dir
    tigerepi.run('ag', r'C:\T1w_dir') # Producing aseg masks with GPU


** Mac and Windows  are supported.**

** Ubuntu (version >18.04)  are supported.**

```
>>tigerepi  c:\data\**\*T1w.nii -o c:\outputdir -b -m -a -d -f
-b: producing extracted brain
-m: producing the brain mask
-a: producing the aseg mask
-w, Producing the white matter parcellation (work in progress)
-f: faster operation with low-resolution models
```

## Citation

* If you use this application, cite the following paper:

1. Kuo CC, Huang TY, “Referenceless correction of EPI distortion with virtual displacement mapping” (2023)

## ASEG43
| Label | Structure              | Label | Structure               |
| ----- | ---------------------- | ----- | ----------------------- |
| 2     | Left Cerebral WM       | 41    | Right Cerebral WM       |
| 3     | Left Cerebral Cortex   | 42    | Right Cerebral Cortex   |
| 4     | Left Lateral Ventricle | 43    | Right Lateral Ventricle |
| 5     | Left Inf Lat Vent      | 44    | Right Inf Lat Vent      |
| 7     | Left Cerebellum WM     | 46    | Right Cerebellum WM     |
| 8     | Left Cerebellum Cortex | 47    | Right Cerebellum Cortex |
| 10    | Left Thalamus          | 49    | Right Thalamus          |
| 11    | Left Caudate           | 50    | Right Caudate           |
| 12    | Left Putamen           | 51    | Right Putamen           |
| 13    | Left Pallidum          | 52    | Right Pallidum          |
| 14    | 3rd Ventricle          | 53    | Right Hippocampus       |
| 15    | 4th Ventricle          | 54    | Right Amygdala          |
| 16    | Brain Stem             | 58    | Right Accumbens area    |
| 17    | Left Hippocampus       | 60    | Right VentralDC         |
| 18    | Left Amygdala          | 62    | Right vessel            |
| 24    | CSF                    | 63    | Right choroid plexus    |
| 26    | Left Accumbens area    | 77    | WM hypointensities      |
| 28    | Left VentralDC         | 85    | Optic Chiasm            |
| 30    | Left vessel            | 251   | CC Posterior            |
| 31    | Left choroid plexus    | 252   | CC Mid Posterior        |
|       |                        | 253   | CC Central              |
|       |                        | 254   | CC Mid Anterior         |
|       |                        | 255   | CC Anterior             |
