# Generative Image Inpainting with Contextual Attention

[Paper](http://jhyu.me/resources/publications/yu2018-generative-inpainting-paper.pdf) | [ArXiv](https://arxiv.org/abs/1801.07892) | [Project](http://jhyu.me/posts/2018/01/20/generative-inpainting.html) | [Demo](http://jhyu.me/posts/2018/01/20/generative-inpainting.html#post)

<img src="https://user-images.githubusercontent.com/22609465/35317673-845730e4-009d-11e8-920e-62ea0a25f776.png" width="425"/> <img src="https://user-images.githubusercontent.com/22609465/35317674-846418ea-009d-11e8-90c7-652e32cef798.png" width="425"/>
<img src="https://user-images.githubusercontent.com/22609465/35317678-848aa3fc-009d-11e8-84a5-01be01a31fc6.png" width="210"/> <img src="https://user-images.githubusercontent.com/22609465/35317679-8496ab84-009d-11e8-945c-e1f957b04288.png" width="210"/>
<img src="https://user-images.githubusercontent.com/22609465/35347783-c5e948fe-00fb-11e8-819c-8212d4edcfd3.png" width="210"/> <img src="https://user-images.githubusercontent.com/22609465/35347784-c5f4242c-00fb-11e8-8e46-5ad224e15096.png" width="210"/>

Example inpainting results of our method on images of natural scene (Places2), face (CelebA) and object (ImageNet). Missing regions are shown in white. In each pair, the left is input image and right is the direct output of our trained generative neural networks without any post-processing.

**Training/testing code and models will be released soon. Please stay tuned.**

## Run train/test

* Install [tensorflow](https://www.tensorflow.org/install/) and tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (`pip install git+https://github.com/JiahuiYu/neuralgym`).  
* Training:
  * Prepare training images filelist.
  * Modify [inpaint.yml](/inpaint.yml) to set `DATA_FLIST`, `LOG_DIR`, `IMG_SHAPES` and other parameters.
  * Run `python train.py`.
* Resume training:
  * Modify `MODEL_RESTORE` flag in [inpaint.yml](/inpaint.yml). E.g., `MODEL_RESTORE: '20180115220926508503_places2_model'`
  * Run `python train.py`.
* Run testing: `python test.py --image examples/input.png --mask examples/mask.png --checkpoint model_logs/your_model_dir`.

## Pretrained models

[Places2]() | [CelebA]() | [CelebA-HQ]() | [ImageNet]()

Download the model dirs and put it under `model_logs/`. Run testing or resume training as described above. All models are trained with images of resolution 256x256 and largest hole size 128x128, above which the results may be deteriorated. We provide several example test cases. Please run:

```bash
# Places2 512x680 input
python test.py --image examples/input.png --mask examples/mask.png --checkpoint model_logs/your_model_dir
# CelebA 256x256 input
python test.py --image examples/input.png --mask examples/mask.png --checkpoint model_logs/your_model_dir
# CelebA-HQ 256x256 input
python test.py --image examples/input.png --mask examples/mask.png --checkpoint model_logs/your_model_dir
# ImageNet 256x256 input
python test.py --image examples/input.png --mask examples/mask.png --checkpoint model_logs/your_model_dir
```

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International


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
```
