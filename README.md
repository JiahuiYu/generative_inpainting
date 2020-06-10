# Generative Image Inpainting

![version](https://img.shields.io/badge/version-v2.0.0-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/tensorflow-v1.7.0-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

An open source framework for generative image inpainting task, with the support of [Contextual Attention](https://arxiv.org/abs/1801.07892) (CVPR 2018) and [Gated Convolution](https://arxiv.org/abs/1806.03589) (ICCV 2019 Oral).

**For the code of previous version (DeepFill v1), please checkout branch [v1.0.0](https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0).**

[CVPR 2018 Paper](https://arxiv.org/abs/1801.07892) | [ICCV 2019 Oral Paper](https://arxiv.org/abs/1806.03589) | [Project](http://jiahuiyu.com/deepfill) | [Demo](http://jiahuiyu.com/deepfill) | [YouTube v1](https://youtu.be/xz1ZvcdhgQ0) | [YouTube v2](https://youtu.be/uZkEi9Y2dj4) | [BibTex](#citing)

<img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case1_raw.png" width="33%"/> <img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case1_input.png" width="33%"/> <img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case1_output.png" width="33%"/>
<img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_raw.png" width="33%"/> <img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_input.png" width="33%"/> <img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_output.png" width="33%"/>

Free-form image inpainting results by our system built on gated convolution. Each triad shows original image, free-form input and our result from left to right.

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

[Places2](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO?usp=sharing) | [CelebA-HQ](https://drive.google.com/drive/folders/1uvcDgMer-4hgWlm6_G9xjvEQGP8neW15?usp=sharing)

Download the model dirs and put it under `model_logs/` (rename `checkpoint.txt` to `checkpoint` because google drive automatically add ext after download). Run testing or resume training as described above. All models are trained with images of resolution 256x256 and largest hole size 128x128, above which the results may be deteriorated. We provide several example test cases. Please run:

```bash
# Places2 512x680 input
python test.py --image examples/places2/case1_input.png --mask examples/places2/case1_mask.png --output examples/places2/case1_output.png --checkpoint_dir model_logs/release_places2_256
# CelebA-HQ 256x256 input
# Please visit CelebA-HQ demo at: jhyu.me/deepfill
```

**Note:** Please make sure the mask file completely cover the masks in input file. You may check it with saving a new image to visualize `cv2.imwrite('new.png', img - mask)`.

## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.

## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purposes only.

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
