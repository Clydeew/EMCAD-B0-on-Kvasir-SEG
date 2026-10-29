# EMCAD-B0 on Kvasir-SEG

This repository is a local adaptation of the [EMCAD authors' code](https://github.com/SLDGroup/EMCAD) at commit `26c9c31`. It contains the Kvasir-SEG preparation and training changes used for a five-seed PVTv2-B0 experiment. It is not the unmodified official repository. The training script is the saved version from the completed experiment; the working copy used afterward was changed to a constant-rate setup.

## Experiment setup

| Item | Setting |
| --- | --- |
| Model | ImageNet-pretrained PVTv2-B0 encoder with EMCAD decoder; encoder and decoder trained together |
| Dataset | Kvasir-SEG, using the authors' fixed 800 train / 100 validation / 100 test lists |
| Input | 352 × 352, with training scales 0.75, 1.0, and 1.25 |
| Training | 200 epochs; batch size 16; AdamW; weight decay 0.0001; gradient clipping 0.5 |
| Learning rate | 0.0005 initially; `CosineAnnealingLR` over 200 epochs, minimum 0.000001 |
| Runs | Five training seeds: 2222–2226, on the same data split |
| Selection | Best checkpoint selected by validation Dice; test evaluated once after each run |
| Tracking | Weights & Biases project `EMCAD-polyp`, plus local text logs |

The learning-rate schedule above matches the **completed five-run experiment** recorded below. `run_kvasir.sh` passes 0.0005 explicitly; the standalone training script has a 0.0001 default when called without that flag. The [EMCAD paper](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html) states a learning rate of 0.0001 and does not specify this schedule. These results should therefore be described as a run of this configuration, not an exact replication of every paper setting.

## Completed run results

These are test-set metrics at the checkpoint with the best validation Dice, recorded on 28 October 2026. They are from the completed reference experiment; **this local checkout has not been trained**. The five text logs are in `logs/`.

| Run | Seed | Test Dice | Test IoU |
| ---: | ---: | ---: | ---: |
| 1 | 2222 | 0.9208 | 0.8685 |
| 2 | 2223 | 0.9110 | 0.8598 |
| 3 | 2224 | 0.9137 | 0.8632 |
| 4 | 2225 | 0.9120 | 0.8591 |
| 5 | 2226 | 0.9236 | 0.8718 |
| **Mean ± sample SD** | | **0.9162 ± 0.0056** | **0.8645 ± 0.0055** |

The table rounds each run to four decimals; the mean and sample standard deviation were calculated from the unrounded W&B summaries. No local checkpoints or Python environment are included.

## Local contents and preparation

`data/polyp/source/kvasir-seg.zip` contains the source dataset. `data/polyp/author_lists/` has the three split lists, and `data/polyp/target/Kvasir/` contains local links to the extracted image/mask pairs. The pretrained B0 encoder weights are in `pretrained_pth/pvt/pvt_v2_b0.pth`. Run `python prepare_kvasir.py` to validate and regenerate the local split links if needed.

The runner expects a Python 3.10 environment at `.venv/`. To create one with `uv` on a machine with a compatible CUDA setup:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements-polyp.txt
```

After setting up Weights & Biases locally, `./run_kvasir.sh` starts five sequential runs with the configuration above. It was **not** run while preparing this local copy.

## Attribution

EMCAD was developed by Md Mostafijur Rahman, Mustafa Munir, and Radu Marculescu. See the [original repository](https://github.com/SLDGroup/EMCAD) and [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html). This adaptation retains the original `LICENSE`.

```bibtex
@inproceedings{rahman2024emcad,
  title={EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation},
  author={Rahman, Md Mostafijur and Munir, Mustafa and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11769--11779},
  year={2024}
}
```
