import sys
import os
from os.path import basename, join, isdir
import argparse
import time
import numpy as np

import glob
import platform
import nibabel as nib

from tigerepi import lib_tool
from tigerepi import lib_seg

from nilearn.image import resample_to_img, reorder_img

def produce_mask(model, f, GPU=False, brainmask_nib=None, index=0):

    model_ff = lib_tool.get_model(model)
    input_nib = nib.load(f)
    input_nib_resp = lib_seg.read_file(model_ff, f)
    
    input_nib_resize = nib.Nifti1Image(lib_tool.ResizeWithPadOrCrop(input_nib_resp.get_fdata(), (240, 240, 180)), input_nib_resp.affine)
    
    mask_nib_resize, prob_resize = lib_seg.run(
        model_ff, input_nib_resize, GPU=GPU, index=index)

    mask_nib_resp = nib.Nifti1Image(lib_tool.ResizeWithPadOrCrop(mask_nib_resize.get_fdata(), input_nib_resp.shape), input_nib_resp.affine)
    
    prob_resp = lib_tool.ResizeWithPadOrCrop(prob_resize, prob_resize.shape[:-3] + input_nib_resp.shape)
    
    mask_nib = resample_to_img(
        mask_nib_resp, input_nib, interpolation="nearest")

    if brainmask_nib is None:
        output = mask_nib.get_fdata()
    else:
        output = mask_nib.get_fdata() * brainmask_nib.get_fdata()
    output = output.astype(int)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)

    return output_nib

def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)


def main():
      
    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output segmentation, default: the directory of input files')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-m', '--betmask', action='store_true', help='Producing bet mask')
    parser.add_argument('-a', '--aseg', action='store_true', help='Producing aseg mask')
    parser.add_argument('-b', '--bet', action='store_true', help='Producing bet images')
    parser.add_argument('-w', '--wmp', action='store_true', help='Producing white matter parcellation')
    parser.add_argument('-k', '--dkt', action='store_true', help='Producing dkt mask')
    parser.add_argument('--index', default=0, type=str, help='The index of the b0 slice, default: 0 (the first slice)')
    parser.add_argument('--model', default=None, type=str, help='Specifies the modelname')
    #parser.add_argument('--report',default='True',type = strtobool, help='Produce additional reports')
    args = parser.parse_args()
    run_args(args)

def seg(argstring, input, output=None, index=0, model=None):

    from argparse import Namespace
    args = Namespace()

    args.betmask = 'm' in argstring
    args.aseg = 'a' in argstring
    args.bet = 'b' in argstring
    args.gpu = 'g' in argstring
    args.wmp = 'w' in argstring
    args.dkt = 'k' in argstring

    if not isinstance(input, list):
        input = [input]
    args.input = input
    args.output = output
    args.index = index
    args.model = model
    run_args(args)   


def run_args(args):

    get_m = args.betmask
    get_a = args.aseg
    get_b = args.bet
    get_w = args.wmp
    get_k = args.dkt

    if True not in [get_m, get_a, get_b, get_w]:
        get_b = True
        # Producing extracted brain by default 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output

    index = int(args.index)
    
    default_model = dict()

    default_model['bet'] = 'epi_bet_v003_r10.onnx'
    default_model['aseg'] = 'epi_aseg43_v003_r10.onnx'
    default_model['wmp'] = 'epi_wmp_v003_r10.onnx'
    default_model['dkt'] = 'epi_dkt_v003_r10.onnx'


    # if you want to use other models
    if isinstance(args.model, dict):
        for mm in args.model.keys():
            default_model[mm] = args.model[mm]
    elif isinstance(args.model, str):
        import ast
        model_dict = ast.literal_eval(args.model)
        for mm in model_dict.keys():
            default_model[mm] = model_dict[mm]

        
    model_bet = default_model['bet']
    model_aseg = default_model['aseg']
    model_wmp = default_model['wmp']
    model_dkt = default_model['dkt']



    print('Total nii files:', len(input_file_list))

    for f in input_file_list:

        print('Processing :', os.path.basename(f))
        t = time.time()

        f_output_dir = output_dir

        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)


        ftemplate = basename(f).replace('.nii', f'_@@@@.nii')
        ftemplate = join(f_output_dir, ftemplate)

        
        tbetmask_nib = produce_mask(model_bet, f, GPU=args.gpu, index=index)
        if get_m:
            save_nib(tbetmask_nib, ftemplate, 'tbetmask')

        if get_b:
            input_nib = nib.load(f)
            if len(input_nib.shape)==4:
                bet = input_nib.get_fdata()[...,index] * tbetmask_nib.get_fdata()
            else:
                bet = input_nib.get_fdata() * tbetmask_nib.get_fdata()

            bet = bet.astype(input_nib.get_data_dtype())


            bet = nib.Nifti1Image(bet, input_nib.affine,
                                  input_nib.header)

            save_nib(bet, ftemplate, 'tbet')
        

        if get_a:
            aseg_nib = produce_mask(model_aseg, f, GPU=args.gpu, index=index, 
                                    brainmask_nib=tbetmask_nib)
            save_nib(aseg_nib, ftemplate, 'aseg')

        
        if get_w:
            wmp_nib = produce_mask(model_wmp, f, GPU=args.gpu, index=index, 
                                    brainmask_nib=tbetmask_nib)
 
            save_nib(wmp_nib, ftemplate, 'wmp')

        if get_k:
            dkt_nib = produce_mask(model_dkt, f, GPU=args.gpu, index=index, 
                                    brainmask_nib=tbetmask_nib)
 
            save_nib(dkt_nib, ftemplate, 'dkt')

        

        print('Processing time: %d seconds' %  (time.time() - t))



if __name__ == "__main__":
    main()
    # if platform.system() == 'Windows':
    #     os.system('pause')

