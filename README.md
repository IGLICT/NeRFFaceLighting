## NeRFFaceLighting: Implicit and Disentangled Face Lighting Representation Leveraging Generative Prior in Neural Radiance Fields <br><sub>Official PyTorch implementation of the ACM Transactions on Graphics paper</sub>

![Teaser image](./docs/teaser.png)

**NeRFFaceLighting: Implicit and Disentangled Face Lighting Representation Leveraging Generative Prior in Neural Radiance Fields**<br>
Kaiwen Jiang, Shu-Yu Chen, Hongbo Fu, Lin Gao<br>

[**Paper**](https://dl.acm.org/doi/10.1145/3597300)

Abstract: *3D-aware portrait lighting control is an emerging and promising domain thanks to the recent advance of generative adversarial networks and neural radiance fields. Existing solutions typically try to decouple the lighting from the geometry and appearance for disentangled control with an explicit lighting representation (e.g., Lambertian or Phong). However, they either are limited to a constrained lighting condition (e.g., directional light) or demand a tricky-to-fetch dataset as supervision for the intrinsic compositions (e.g., the albedo). We propose NeRFFaceLighting to explore an implicit representation
for portrait lighting based on the pretrained tri-plane representation to address the above limitations. We approach this disentangled lighting-control problem by distilling the shading from the original fused representation of both appearance and lighting (i.e., one tri-plane) to their disentangled representations (i.e., two tri-planes) with the conditional discriminator to supervise the lighting effects. We further carefully design the regularization to reduce the ambiguity of such decomposition and enhance the ability of generalization to unseen lighting conditions. Moreover, our method can be extended to enable 3D-aware real portrait relighting. Through extensive quantitative and qualitative evaluations, we demonstrate the superior 3D-
aware lighting control ability of our model compared to alternative and existing solutions.*

## Requirements

* We have done all training, testing and development using V100 GPUs on the Linux platform. We recommend 1 high-end NVIDIA GPU for testing, and 4+ high-end NVIDIA GPUs for training.
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later. (Why is a separate CUDA toolkit installation required?  We use the custom CUDA extensions from the StyleGAN3 repo. Please see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml`
  - `conda activate eg3d`
  - `python -m ipykernel install --user --name=eg3d`

## Getting started

- Please download necessary models following the [link](https://drive.google.com/drive/folders/1MT1aZJa0GEblJv4YUyVNi0BdwgGnQB_I?usp=sharing) and put it under `./data` for training and demonstrations. 
- Since our model relies on other repos to train and evaluate, we provide our modified source codes under the directory `external_dependencies` (most changes originate from reducing the package dependencies).
- Additionally, please:
  - Go to the [link](https://github.com/yfeng95/DECA/tree/master) to download its `data` directory, execute `bash fetch_data.sh`, and put its supplemented `data` directory under `./eg3d/external_dependencies/`.
  - Go to the [link](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/master) to download its `BFM` directory, follow the `Prepare prerequisite models` section (we set 'model_name' as 'pretrained' ), and put its supplemented `BFM` and `checkpoints` directories under `./eg3d/external_dependencies/deep3drecon/` and duplicate copies (or create symbol links) under `./projector/modules/`.
  - Go to the [link](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh) to download `backbone_ir50_ms1m_epoch120.pth`, and put it under `./eg3d/external_dependencies/face_evolve/`.
  - Go to the [link](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) to download `79999_iter.pth`, and put it under `./eg3d/external_dependencies/face_parsing/`.

## Generating media & Evaluation

We provide a notebook `./eg3d/demo.ipynb` for demonstrating how to generate a sample, perform relighting, export the geometry, and how to evaluate the lighting error, lighting error (unseen), lighting stability, and FID values.

We provide a notebook `./projector/demo.ipynb` for demonstrating how to project a real portrait into our model, and perform relighting.

**NOTICE**: Please read through the notebooks before you get started with our model. Necessary noticing messages are provided along with the examples.

## Preparing datasets

**FFHQ**: please refer to the [EG3D](https://github.com/NVlabs/eg3d) repository for processing, or downloading the recropped FFHQ dataset (The images are supposed to be moved out of the sub-directories). After finishing, please use the `./data/dataset.json` to replace the original `dataset.json`.

**CelebA-HQ**(Optional): we use such a dataset for evaluating the metrics 'lighting error (unseen)', which can be downloaded from [here](https://github.com/tkarras/progressive_growing_of_gans). Notice that downloading the raw images are enough since the processing scripts are embedded in the notebook.

Additional NOTE:
- If you want to use your own dataset, you should first crop the portraits, and then use the *Deep Single Portrait Relighting* to extract the lighting condition as spherical harmonics from the portraits. Please refer to the `dataset.json` to see how labels are arranged.

- For people who want to extend to other domains other than human faces, you have better find a robust model to estimate the lighting conditions for, say, cat faces. :)

## Training

Please remember to **update some paths** before training.

You can use the following command with 4+ high-end NVIDIA GPUs to train the model:
```bash
$ conda activate eg3d
$ cd eg3d && chmod +x ./start.sh
$ ./start.sh
```
NOTE: Checkpoints at `004000~005000` kimg are basically fine.

You can use the following command with 1 high-end NVIDIA GPU to train the encoder:
```bash
$ conda activate eg3d
$ cd encoder4editing && chmod +x ./start.sh
$ ./start.sh
```

## Citation
```
@article{nerffacelighting,
  author  = {Jiang, Kaiwen and Chen, Shu-Yu and Fu, Hongbo and Gao, Lin},
  title   = {NeRFFaceLighting: Implicit and Disentangled Face Lighting Representation Leveraging Generative Prior in Neural Radiance Fields},
  year    = {2023},
  journal = {ACM Transactions on Graphics (TOG)}
}
```

## Acknowledgements
This repository relies on the [EG3D](https://github.com/NVlabs/eg3d), [DECA](https://github.com/yfeng95/DECA), [Deep3DRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [DPR](https://github.com/zhhoper/DPR), [E4E](https://github.com/omertov/encoder4editing), [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe), [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch/tree/master), and [facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch).
