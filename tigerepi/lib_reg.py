import os
import ants
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import onnxruntime as ort
from tigerepi import lib_tool
from scipy.ndimage import map_coordinates


def affine(input_dir, output_dir, fixed_image_path=None, fixed_antspy_image=None, b0_file=None, do_write=False):
    
    if fixed_antspy_image is not None:
        fixed_image = fixed_antspy_image
    else:
        fixed_image = ants.image_read(fixed_image_path)

    b0_path = os.path.join(input_dir, b0_file)
    if not os.path.isfile(b0_path):
        raise FileNotFoundError(f"B0 file not found: {b0_path}")
    moving_b0 = ants.image_read(b0_path)
    print(f'b0影像為{b0_path}')
    
    tx = ants.registration(
        fixed=fixed_image,
        moving=moving_b0,
        type_of_transform='Affine'
    )['fwdtransforms']

    warped_dict = {}

    for filename in os.listdir(input_dir):
        if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            continue

        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        if do_write and os.path.exists(dst_path):
            print(f'檔案 {dst_path} 已存在，跳過')
            continue

        img = ants.image_read(src_path)
        warped = ants.apply_transforms(
            fixed=fixed_image,
            moving=img,
            transformlist=tx,
            interpolator='linear'
            )
        
        if do_write:
            ants.image_write(warped, dst_path)
        else:
            warped_dict[filename] = warped

    if do_write:
        print(f"Registration & transform completed. Outputs in: {output_dir}")
        #return warped_dict, tx
    else:
        return warped_dict, tx


def norm_array(data: np.ndarray):
    
    foreground_mask = data != 0

    if not np.any(foreground_mask):
        return np.zeros_like(data)

    fg_values = data[foreground_mask]
    data_min = fg_values.min()
    data_max = fg_values.max()

    normalized_data = np.zeros_like(data, dtype=np.float32)

    if data_max > data_min:
        normalized_data[foreground_mask] = (fg_values - data_min) / (data_max - data_min)
    else:
        normalized_data[foreground_mask] = data_min

    return normalized_data


def resample_nifti_image(nib_img: nib.Nifti1Image, voxel_sizes: tuple = (1.0, 1.0, 1.0), interpolation_order: int = 1):
    
    try:
        resampled = resample_to_output(nib_img, voxel_sizes=voxel_sizes, order=interpolation_order)
    except Exception as e:
        raise RuntimeError(f"[Error] resample_to_output 失敗: {e}")

    return resampled

    
def pad_and_crop(nib_img: nib.Nifti1Image, target_shape: tuple):
   
    data = nib_img.get_fdata()
    original_affine = nib_img.affine.copy()
    header = nib_img.header.copy()
    target_np_shape = (target_shape[2], target_shape[1], target_shape[0])
    
    def pad_to_shape(img, target_np_shape):
        current_shape = img.shape
        padding_amount = [max(0, t - s) for s, t in zip(current_shape, target_np_shape)]
        pad_width = [(p // 2, p - (p // 2)) for p in padding_amount]
        padded = np.pad(img, pad_width, mode='constant', constant_values=0)
        return padded, pad_width

    def adjust_affine_for_padding(original_affine, pad_width):
        
        shift_vox = [pad[0] for pad in pad_width]
        new_aff = original_affine.copy()
        new_aff[:3, 3] -= np.dot(original_affine[:3, :3], shift_vox)
        return new_aff

    def crop_image(img, target_np_shape):
        current_shape = img.shape
        crop_slices = []
        for i in range(3):
            start = (current_shape[i] - target_np_shape[i]) // 2
            end = start + target_np_shape[i]
            crop_slices.append(slice(start, end))
        cropped = img[tuple(crop_slices)]
        return cropped, crop_slices

    def update_affine_after_crop(original_affine, crop_slices):
        translation_vox = [s.start for s in crop_slices]
        new_aff = original_affine.copy()
        new_aff[:3, 3] += np.dot(original_affine[:3, :3], translation_vox)
        return new_aff

    padded_data, pad_width = pad_to_shape(data, target_np_shape)
    affine_after_pad = adjust_affine_for_padding(original_affine, pad_width)
    cropped_data, crop_slices = crop_image(padded_data, target_np_shape)
    final_affine = update_affine_after_crop(affine_after_pad, crop_slices)
    
    new_img = nib.Nifti1Image(cropped_data.astype(data.dtype), final_affine, header=header)
    return new_img


def predict(model, data, GPU, mode=None):

    so = ort.SessionOptions()
    cpu = max(int(lib_tool.cpu_count()*0.7), 1)
    so.intra_op_num_threads = cpu
    so.inter_op_num_threads = cpu
    so.log_severity_level = 3

    if GPU and (ort.get_device() == "GPU"):
        session = ort.InferenceSession(model,
                                       providers=['CUDAExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)

    data_type = 'float64'
    if mode == 'reg':
        input_names = [input.name for input in session.get_inputs()]
        inputs = {input_names[0]: data[0], input_names[1]: data[1]}
        return session.run(None, inputs)
    return session.run(None, {session.get_inputs()[0].name: data.astype(data_type)}, )[0]

def apply_displacement_field(moving_np3d, disp_field):
    
    D, H, W = moving_np3d.shape
    zi, yi, xi = np.meshgrid(
        np.arange(D),
        np.arange(H),
        np.arange(W),
        indexing='ij'
    )
    
    new_z = zi + disp_field[..., 0]
    new_y = yi + disp_field[..., 1]
    new_x = xi + disp_field[..., 2]

    coords = [
        new_z.flatten(),
        new_y.flatten(),
        new_x.flatten()
    ]
    warped_flat = map_coordinates(
        moving_np3d,
        coords,
        order=1,
        mode='nearest'
    )
    warped_np3d = warped_flat.reshape(D, H, W)
    return warped_np3d

    
def ants_image_to_nib_affine(ants_img):
    
    spacing = np.array(ants_img.spacing)
    origin  = np.array(ants_img.origin)
    R_lps   = np.array(ants_img.direction).reshape(3, 3)
    
    flip = np.diag([-1, -1, 1])
    R_ras = flip @ R_lps
    M_ras = R_ras @ np.diag(spacing)
    origin_ras = flip @ origin
    
    affine_ras = np.eye(4, dtype=np.float32)
    affine_ras[:3, :3] = M_ras
    affine_ras[:3,  3] = origin_ras
    
    return affine_ras

