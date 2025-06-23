import argparse
import glob
from os.path import join
import ants
import tempfile
import nibabel as nib
import numpy as np
import os
from nilearn.image import reorder_img

from tigerepi import lib_tool
from tigerepi import lib_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
    parser.add_argument('-b0', '--b0_index', default=None, type=str, help='The index of b0 slice or the .bval file, default: 0 (the first slice)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')    
    parser.add_argument('-f', '--fixed', default=os.path.join('template', 'MNI152_T1_1mm_brain.nii.gz'), help="fixed image path (default: template/MNI152_T1_1mm_brain.nii.gz)")
    parser.add_argument('-a', '--ants', action='store_true', help="affine only")

    args = parser.parse_args()
    run_args(args)

def reg(input, output=None, fixed=None, b0_index=0, GPU=False, ants=False):

    from argparse import Namespace
    args = Namespace()

    args.b0_index = str(b0_index)
    args.gpu = GPU
    args.ants  = ants
    args.fixed = fixed

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output

    run_args(args)  

def run_args(args):

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))
    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])
    print('Total nii files:', len(input_file_list))
    
    if args.b0_index is None:
        b0_index = 0
    elif os.path.exists(args.b0_index.replace('.bval', '') + '.bval'):
        b0_index = lib_vdm.get_b0_slice(args.b0_index.replace('.bval', '') + '.bval')
    else:
        b0_index = int(args.b0_index)
        
    first_input = args.input[0]
    if os.path.isdir(first_input):
        input_dir = first_input
    else:
        input_dir = os.path.dirname(os.path.abspath(input_file_list[0]))

    if args.output is None:
        output_dir = input_dir
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    try:
        b0_filepath = input_file_list[b0_index]
    except IndexError:
        raise IndexError(f"b0_index={b0_index} over the input_file_list range (total {len(input_file_list)} 張影像)。")
    b0_filename = os.path.basename(b0_filepath)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_template = os.path.join(this_dir, 'template', 'MNI152_T1_1mm_brain.nii.gz')
    
    if args.fixed is None or args.fixed == default_template:
        warped_dict =  lib_reg.affine(
        input_dir,
        output_dir,
        default_template,
        fixed_antspy_image=None,
        b0_file=b0_filename,
        do_write=args.ants
        )
        fixed_for_norm = ants.image_read(default_template)
    else:
        fixed_image_path = args.fixed
    
        reordered_fixedimg = reorder_img(fixed_image_path, resample='continuous')
        resampled_fixed_nib = lib_reg.resample_nifti_image(reordered_fixedimg,  voxel_sizes=(1.0, 1.0, 1.0), interpolation_order=1)
        final_fixed_nib = lib_reg.pad_and_crop(resampled_fixed_nib, (192, 224, 160))
        
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            nib.save(final_fixed_nib, tmp.name)
            antspy_fixed_for_affine = ants.image_read(tmp.name)
                    
            warped_dict = lib_reg.affine(
            input_dir,
            output_dir,
            fixed_image_path=None,
            fixed_antspy_image=antspy_fixed_for_affine,
            b0_file=b0_filename,
            do_write=args.ants
            )
        fixed_for_norm = antspy_fixed_for_affine
            
    if args.ants is True:
        print("only do affine reg")
        return
    
    warped_b0_image = warped_dict.get(b0_filename)
    warped_b0_np = warped_b0_image.numpy()
    norm_b0 = lib_reg.norm_array(warped_b0_np)

    fixed_np = fixed_for_norm.numpy()
    norm_fixed = lib_reg.norm_array(fixed_np)
    
    input_b0   = norm_b0.astype(np.float32)[None, ...][None, ...]
    input_fixed = norm_fixed.astype(np.float32)[None, ...][None, ...]

    model_path = lib_tool.get_model('epireg_unet3d_v2.0')
    
    disp_field_list = lib_reg.predict(model_path, [input_b0, input_fixed], args.gpu, mode='reg')
    raw_disp = disp_field_list[1]
    disp = np.squeeze(raw_disp)
    
    if disp.ndim == 4 and disp.shape[0] == 3:
        disp = np.stack([disp[0], disp[1], disp[2]], axis=-1)
    
    assert disp.ndim == 4 and disp.shape[-1] == 3

    for filename, affine_img in warped_dict.items():
        moving_np3d = affine_img.numpy().astype(np.float32)
        warped_np3d = lib_reg.apply_displacement_field(moving_np3d, disp)
        out_affine = lib_reg.ants_image_to_nib_affine(affine_img)

        warped_nib = nib.Nifti1Image(warped_np3d, out_affine)

        if filename.endswith('.nii.gz'):
            out_fname = filename[:-7] + '_reg.nii.gz'
        elif filename.endswith('.nii'):
            out_fname = filename[:-4] + '_reg.nii'  
        out_path  = os.path.join(output_dir, out_fname)
        nib.save(warped_nib, out_path)

        print(f"{filename} , output → {out_fname}")

    return

if __name__ == "__main__":
    main()
