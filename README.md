# PooDLe: Pooled and dense self-supervised learning from naturalistic videos

### [Paper](https://arxiv.org/abs/2408.11208) | [Website](https://agenticlearning.ai/poodle)

This project hosts the code for implementing the PooDLe framework for self-supervised learning from videos.

> [**PooDLe: Pooled and dense self-supervised learning from naturalistic videos**](https://arxiv.org/abs/2408.11208)<br>
> [Alex N. Wang*](https://www.alexn.wang/), [Christopher Hoang*](https://www.chrishoang.com/), [Yuwen Xiong](https://www.cs.toronto.edu/~yuwen), [Yann LeCun](https://yann.lecun.com/), [Mengye Ren](https://mengyeren.com/)<br>
> *arXiv preprint ([arXiv 2408.11208](https://arxiv.org/abs/2408.11208))*

## Pretrained models

<table>
  <tr>
    <th colspan="1">model</th>
    <th colspan="1">resolution</th>
    <th colspan="1">epochs</th>
    <th colspan="1">data</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>PooDLe</td>
    <td>512x1024</td>
    <td>100</td>
    <td>BDD100K</td>
    <td><a href="https://drive.google.com/file/d/1PHQtzxuJNn5cAMQfsxyKXlMSsTKWZpds/view">full checkpoint</a></td>
    <td><a href="configs/exp/poodle.yaml">configs</a></td>
  </tr>
  <tr>
    <td>PooDLe</td>
    <td>512x1024</td>
    <td>10</td>
    <td>Walking Tours Venice</td>
    <td><a href="https://drive.google.com/file/d/14aGObDItps8cE9VOpt2KHPliINKrGl7v/view">full checkpoint</a></td>
    <td><a href="configs/exp/poodle_wt.yaml">configs</a></td>
  </tr>
  <tr>
    <td>PooDLe</td>
    <td>512x1024</td>
    <td>20</td>
    <td>Walking Tours All</td>
    <td><a href="https://drive.google.com/file/d/18Eh3rdr0dDi5e0RPz-Oumfvy1JR0Z4RW/view">full checkpoint</a></td>
    <td><a href="configs/exp/poodle_wt.yaml">configs</a></td>
  </tr>
  <tr>
    <td>FlowE</td>
    <td>512x1024</td>
    <td>100</td>
    <td>BDD100K</td>
    <td><a href="https://drive.google.com/file/d/1ZCXijz0qin99L4l7qS2bE3ldg70JuDiS/view">full checkpoint</a></td>
    <td><a href="configs/exp/flowe.yaml">configs</a></td>
  </tr>
</table>

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   main training loop for poodle
│   ├── train_uflow.py        #   main training loop for unsupervised flow model
│   ├── datasets              #   datasets, data loaders
│   ├── models                #   model definitions
│   ├── routines              #   additional training routines
│   └── utils                 #   shared utilities
└── main.py         # entrypoint for launch PooDLe pre-training locally or SLURM cluster
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.

## Launching PooDLe pre-training
[main.py](main.py) is an entrypoint script for launching experiments with [submitit](https://github.com/facebookincubator/submitit) and [hydra](https://hydra.cc).
The actual implementation is in [src/train.py](src/train.py), which parses the experiment config file and runs the PooDLe pre-training.

### Training
Here is an example of how to run ablation-sized PooDLe pre-training on a local, 2 GPU machine with config [configs/exp/poodle_ablation.yaml](configs/exp/poodle_ablation.yaml):

```
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
exp=poodle_ablation \
name='poodle-ablation-bdd100k'
```

*Note: This example is just for illustrative purposes. The full PooDLe config should be run for an effective batch-size of 128, in order to reproduce our results.*

<!-- ### SLURM cluster training
Here is an example of how to run full PooDLe pre-training on a SLURM cluster with 32 GPUs with config [configs/poodle.yaml](configs/poodle.yaml):
```
python main.py \
compute=slurm compute/cluster=4x8 \
exp=poodle \
name='poodle-full-bdd100k
``` -->

## Evaluation

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) to evaluate on semantic segmentation and [MMDetection](https://github.com/open-mmlab/mmdetection) to evaluate on object detection.

Their tools will work out-of-the-box for UperNet and ResNet encoder-only evaluations. A custom model file must be used running linear evaluations with the SDM as the architecture changes.

---

## Pretraining with unsupervised flow model
We train a UFlow-based model with a PWC backbone for our ablation experiments. The model is first trained on KITTI data using the command
```
python main.py exp=uflow_pwc name='uflow_pwc-kitti'
```

Then we further train it on BDD 
```
python exp=uflow_pwc_bdd name='uflow_pwc-kitti-bdd' \
    occ_start_epochs=0 selfsup_start_epochs=0 selfsup_warmup_epochs=1 \
    warmup_epochs=20 \
    lr_scheduler=cosine \
    resume='"PATH-TO-YOUR-UFLOW-CKPT"'
```

Following, you can train a PooDLe with this flow model with the following command
```
torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py \
  exp=poodle_ablation \
  name='poodle_uflow-ablation-bdd100k' \
  model=poodle_uflow \
  +model_configs.flow_model_checkpoint_uflow='"PATH-TO-YOUR-UFLOW-CKPT"'
```

---

### Requirements
* Python 3.10 (or newer)
* PyTorch 2.2.0
* torchvision 0.17.1 ([build from source, for video_reader](https://github.com/pytorch/vision?tab=readme-ov-file#unstable-video-backend))
* ffmpeg 5.1.2 (from conda-forge, for video_reader)
* [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) (build from source, only needed for unsupervised flow model)
* Other dependencies: decord, ffprobe-python, flow-vis, hydra-core, kornia, numpy, scipy, timm==0.3.2, wandb

Importing this version of `timm` will raise an import error, see [here](https://github.com/huggingface/pytorch-image-models/issues/420) for a fix.

--- 

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation:
```
@article{2024poodle,
  title={PooDLe: Pooled and dense self-supervised learning from naturalistic videos}, 
    author={Alex N. Wang and Chris Hoang and Yuwen Xiong and Yann LeCun and Mengye Ren},
  journal={arXiv preprint arXiv:2408.11208},
  year={2024}
}
