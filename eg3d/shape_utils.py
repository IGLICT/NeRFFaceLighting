# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Utils for extracting 3D shapes using marching cubes. Based on code from DeepSDF (Park et al.)

Takes as input an .mrc file and extracts a mesh.

Ex.
    python shape_utils.py my_shape.mrc
Ex.
    python shape_utils.py myshapes_directory --level=12
"""


import time
import plyfile
import glob
import numpy as np
import os
import torch
import torch.utils.data
from skimage import measure
import argparse
import mrcfile
from tqdm import tqdm

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

@torch.no_grad()
def extract_mrc(
    fn: str, 
    G: torch.nn.Module, 
    w: torch.Tensor, 
    shape_res: int = 512, 
    truncation_psi: float = 0.5, 
    truncation_cutoff: int = 14, 
):
    max_batch=1000000
    samples, voxel_origin, voxel_size = create_samples(
        N=shape_res, 
        voxel_origin=[0, 0, 0], 
        cube_length=G.rendering_kwargs['box_warp'] * 1
    )#.reshape(1, -1, 3)
    samples = samples.to(w.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=w.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=w.device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample_mixed(
                    samples[:, head:head+max_batch], 
                    transformed_ray_directions_expanded[:, :samples.shape[1]-head], 
                    w, 
                    truncation_psi=truncation_psi, 
                    truncation_cutoff=truncation_cutoff, 
                    noise_mode='const'
                )['sigma']
                sigmas[:, head:head+max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value
    
    with mrcfile.new_mmap(fn, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = sigmas

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")


def convert_mrc(input_filename, output_filename, isosurface_level=1):
    with mrcfile.open(input_filename) as mrc:
        convert_sdf_samples_to_ply(np.transpose(mrc.data, (2, 1, 0)), [0, 0, 0], 1, output_filename, level=isosurface_level)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mrc_path')
    parser.add_argument('--level', type=float, default=10, help="The isosurface level for marching cubes")
    args = parser.parse_args()

    if os.path.isfile(args.input_mrc_path) and args.input_mrc_path.split('.')[-1] == 'ply':
        output_obj_path = args.input_mrc_path.split('.mrc')[0] + '.ply'
        convert_mrc(args.input_mrc_path, output_obj_path, isosurface_level=1)

        print(f"{time.time() - start_time:02f} s")
    else:
        assert os.path.isdir(args.input_mrc_path)

        for mrc_path in tqdm(glob.glob(os.path.join(args.input_mrc_path, '*.mrc'))):
            output_obj_path = mrc_path.split('.mrc')[0] + '.ply'
            convert_mrc(mrc_path, output_obj_path, isosurface_level=args.level)