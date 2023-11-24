# SEGSID

Official implementation of "SEGSID: A Semantic-Guided Framework for Sonar Image Despeckling".


---

## Setup

### Requirements

Our experiments are done with:  

- Python 3.9.15
- PyTorch 1.12.1
- numpy 1.23.5
- opencv 4.7.0.72
- scikit-image 0.19.3
- cudatoolkit 11.3.1


### Data Preprocessing
We follow the folder structure and dataset setup in the AP-BSN. Please click [this link](https://github.com/wooseoklee4/AP-BSN#directory) for detailed preparation description.


### Pre-trained Models

You can download pretrained checkpoints of our method. Place these files into `ckpt` folder.

| Method |      Dataset   |      Config file                | Pre-trained |
| :----: | :------------: | :-----------------------------: | :---------: |
| SEGSID |       KLSG     |     ./config/KLSG/config.yaml   | [SEGSID_KLSG.pth](https://drive.google.com/file/d/1WKaCIIJtu4STz5yeHebEb5PoEOFh_7zD/view?usp=drive_link) |
| SEGSID |       URPC     |     ./config/URPC/config.yaml   | [SEGSID_URPC.pth](https://drive.google.com/file/d/1-EvWVIduPLvA0SppzAGO9SCgvykjhTiT/view?usp=drive_link) |
| SEGSID |       DEBRIS   |   ./config/DEBRIS/config.yaml   | [SEGSID_DEBRIS.pth](https://drive.google.com/file/d/1VUSan8gN3KBJluHOOXKZpj0sky03Pzyz/view?usp=drive_link) |
| SEGSID-KD |       KLSG   |    ——                          | [SEGSID_KD_KLSG.pt](https://drive.google.com/file/d/18kPl34PGY1ap0eMjp-bf1bBf_12ZwPSw/view?usp=drive_link) |
| SEGSID-KD |       URPC   |    ——                          | [SEGSID_KD_URPC.pt](https://drive.google.com/file/d/1LnqZbhNdhtvtutnHigAMgACqKFDnBPeT/view?usp=drive_link) |
| SEGSID-KD |     DEBRIS   |   ——                           | [SEGSID_KD_DEBRIS.pt](https://drive.google.com/file/d/12R--_MLvTgHGqvm_PMHG00OByOK4uT09/view?usp=drive_link) |



---

## Quick test

To test sonar images in `demo` folder with well-trained SEGSID in gpu:0. The denoised results are saved in the `./output/demo` folder by default.
```
python test.py --session_name Test_SEGSID_Demo --config KLSG/config --pretrained SEGSID_KLSG.pth --gpu 0 --test_dir ./demo
```

To test sonar images in `demo` folder with fine-tuned SEGSID-KD in gpu:0. The denoised results are saved in the `./output/SEGSID_KD_demo` folder.
```
python test_KD.py --test_dir ./demo/ --model_weight_dir ./ckpt/SEGSID_KD_KLSG.pt --output_dir ./output/SEGSID_KD_demo/
```

Additionally, you can also put the sonar images you need to denoise under the `demo` folder.




---

## Training & Test

### Training 

When training SEGSID, you can control detail experimental configurations (e.g. training loss, epoch, batch_size, etc.) in each of config file. For example:

```
# Train SEGSID on the KLSG dataset using gpu:0
python train.py --session_name train_SEGSID_KLSG --config KLSG/config --gpu 0
```

Before training SEGSID-KD, please make sure you have trained SEGSID on your dataset and got the denoised results from well-trained SEGSID. For example:

```
# Pre-train SEGSID-KD (i.e. the first stage of SEGSID-KD) on the KLSG dataset using gpu:0
python train_KD.py --exp_name KLSG_stage1 --stage 1 --input_dir ./dataset/prep/KLSG_Train/RN/ --lr 3e-4 --lr_min 5e-5

# Fine-tune SEGSID-KD (i.e. the second stage of SEGSID-KD) on the KLSG dataset using gpu:0. 
# The pre-trained model at the first stage is ./ckpt/KD_stage1_KLSG.pt 
python train_KD.py --exp_name KLSG_stage2 --stage 2 --input_dir ./dataset/prep/KLSG_Train/RN/ --label_dir ./dataset/prep/KLSG_Train/RN_DN/ --pretrain_model_dir ./ckpt/KD_stage1_KLSG.pt --lr 1e-4 --lr_min 1e-5
```


### Test


Examples:

```
# Test KLSG dataset with well-trained SEGSID (./ckpt/SEGSID_KLSG.pth) in gpu:0
python test.py --session_name Test_SEGSID_KLSG --config KLSG/config --pretrained SEGSID_KLSG.pth --gpu 0

# Test KLSG dataset with fine-tuned SEGSID-KD (./ckpt/SEGSID_KLSG.pth) in gpu:0
python test_KD.py --test_dir ./dataset/KLSG/test_dataset/ --model_weight_dir ./ckpt/KD_KLSG.pt --output_dir ./output/check_kd_klsg/
```



## Acknowledgement

The codes are based on [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [VQ-VAE-2](https://github.com/rosinality/vq-vae-2-pytorch). Thanks for their awesome works.


