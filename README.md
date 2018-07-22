# Generative Image Inpainting with Contextual Attention

[CVPR 2018 Paper](https://arxiv.org/abs/1801.07892) | [ArXiv](https://arxiv.org/abs/1801.07892) | [Project](http://jiahuiyu.com/deepfill) | [Demo](http://jiahuiyu.com/deepfill) | [YouTube](https://youtu.be/xz1ZvcdhgQ0) | [BibTex](#citing)

**Update (Jun, 2018)**:
1. The tech report of our new image inpainting system DeepFillv2 is released. [ArXiv](http://arxiv.org/abs/1806.03589) | [Project](http://jiahuiyu.com/deepfill2)
2. We also released recorded demo video [YouTube](https://youtu.be/xz1ZvcdhgQ0) based on DeepFillv1 (CVPR 2018), as well as video [YouTube](https://youtu.be/uZkEi9Y2dj4) of DeepFillv2. Best viewed with highest resolution 1080p.
3. DeepFillv1 is trained and mainly works on rectangular masks, while DeepFillv2 can complete images on free-form masks with user guidance as an option.


<img src="https://user-images.githubusercontent.com/22609465/35317673-845730e4-009d-11e8-920e-62ea0a25f776.png" width="425"/> <img src="https://user-images.githubusercontent.com/22609465/35317674-846418ea-009d-11e8-90c7-652e32cef798.png" width="425"/>
<img src="https://user-images.githubusercontent.com/22609465/35317678-848aa3fc-009d-11e8-84a5-01be01a31fc6.png" width="210"/> <img src="https://user-images.githubusercontent.com/22609465/35317679-8496ab84-009d-11e8-945c-e1f957b04288.png" width="210"/>
<img src="https://user-images.githubusercontent.com/22609465/35347783-c5e948fe-00fb-11e8-819c-8212d4edcfd3.png" width="210"/> <img src="https://user-images.githubusercontent.com/22609465/35347784-c5f4242c-00fb-11e8-8e46-5ad224e15096.png" width="210"/>

Example inpainting results of our method on images of natural scene (Places2), face (CelebA) and object (ImageNet). Missing regions are shown in white. In each pair, the left is input image and right is the direct output of our trained generative neural networks without any post-processing.

## Run

0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`).
1. Training:
    * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)). 
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.png --mask examples/mask.png --output examples/output.png --checkpoint model_logs/your_model_dir`.
4. Still have questions?
    * If you still have questions (e.g.: How filelist looks like? How to use multi-gpus? How to do batch testing?), please first search over closed issues. If the problem is not solved, please open a new issue.

## Pretrained models

[CelebA-HQ](https://drive.google.com/open?id=1lpluFXyWDxTY6wcjixQGWX8jxUUMlyBW) | [Places2](https://drive.google.com/open?id=1M3AFy7x9DqXaI-fINSynW7FJSXYROfv-) | [CelebA](https://drive.google.com/open?id=1sP8ViF3mxUMN--xpKqonEeW9d8S8pJEo) | [ImageNet](https://drive.google.com/open?id=136APWSdPRAF7-XoS8sMBTLV-X3f-ogE0) | 

Download the model dirs and put it under `model_logs/` (rename `checkpoint.txt` to `checkpoint` because google drive automatically add ext after download). Run testing or resume training as described above. All models are trained with images of resolution 256x256 and largest hole size 128x128, above which the results may be deteriorated. We provide several example test cases. Please run:

```bash
# Places2 512x680 input
python test.py --image examples/places2/wooden_input.png --mask examples/places2/wooden_mask.png --output examples/output.png --checkpoint_dir model_logs/release_places2_256
# CelebA 256x256 input
python test.py --image examples/celeba/celebahr_patches_164036_input.png --mask examples/center_mask_256.png --output examples/output.png --checkpoint_dir model_logs/release_celeba_256/
# CelebA-HQ 256x256 input
# Please visit CelebA-HQ demo at: jhyu.me/demo
# ImageNet 256x256 input
python test.py --image examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png --mask examples/center_mask_256.png --output examples/output.png --checkpoint_dir model_logs/release_imagenet_256
```

**Note:** Please make sure the mask file completely cover the masks in input file. You may check it with saving a new image to visualize `cv2.imwrite('new.png', img - mask)`.

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.

## FAQ


* Can other types of GANs work in current setting?

The proposed contextual attention module is independent of GAN losses. We experimented with other types of GANs in Section 5.1 and WGAN-GP is used by default. Intuitively, during training, pixel-wise reconstruction loss directly regresses holes to the current ground truth image, while WGANs implicitly learn to match potentially correct images and supervise the generator with adversarial gradients. Both WGANs and reconstruction loss measure image distance with pixel-wise L1.

* How to determine if G is converged?

We use a slightly modified WGAN-GP for adversarial loss. WGANs are demonstrated to have more meaningful convergent curves than others, which is also confirmed in our experiments.

* The split of train/test data in experiments.

We use the default training/validation data split from Places2 and CelebA. For CelebA, training/validation have no identity overlap. For DTD texture dataset which has no default split, we sample 30 images for validation.

* How does random mask generated?

Please check out function `random_bbox` and `bbox2mask` in file [inpaint_ops.py](/inpaint_ops.py).

* Parameters to handle the memory overhead.

Please check out function `contextual_attention` in file [inpaint_ops.py](/inpaint_ops.py).

* How to implement contextual attention?

The proposed contextual attention learns where to borrow or copy feature information from known background patches to reconstruct missing patches. It is implemented with TensorFlow conv2d, extract_image_patches and conv2d_transpose API in file [inpaint_ops.py](/inpaint_ops.py). To test the implementation, one can simply apply it on RGB feature space, using image A to reconstruct image B. This special case then turns into a naive style transfer.

<img src="https://user-images.githubusercontent.com/22609465/36634042-4168652a-1964-11e8-90a9-2c480b97eff7.jpg" height="150"/> <img src="https://user-images.githubusercontent.com/22609465/36634043-4178580e-1964-11e8-9ebf-69c4b6ad52a5.png" height="150"/> <img src="https://user-images.githubusercontent.com/22609465/36634040-413ee394-1964-11e8-8d23-f86a018edf01.png" height="150"/>

```bash
python inpaint_ops.py --imageA examples/style_transfer/bnw_butterfly.png  --imageB examples/style_transfer/bike.jpg --imageOut examples/style_transfer/bike_style_out.png
```


## Citing
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
