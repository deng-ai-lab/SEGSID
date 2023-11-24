from kd.vqvae import VQVAE
import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",           type=str, default=None,     help="Directory of images to denoise.")
    parser.add_argument("--model_weight_dir",   type=str, default=None,     help="Explicit directory of pre-trained model")
    parser.add_argument("--output_dir",         type=str, default=None,     help="Directory of images have been denoised.")
    parser.add_argument("--gpu_id",             type=str, default='0',      help="(optional)  GPU ID(number). Only support single gpu setting.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print('the result is saved in ', args.output_dir)

    device = f'cuda:{args.gpu_id}'
    model = VQVAE(in_channel=1).to(device)
    checkpoint = torch.load(args.model_weight_dir, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    for root, dirs, files in os.walk(args.test_dir):
        img_paths = files
        img_paths.sort()
        break

    with torch.no_grad():
        time_begin = time.time()
        for idx, img_path in enumerate(img_paths):
            # print(idx, img_path)

            """ read images """
            img = cv2.imread(os.path.join(args.test_dir, img_path), 0)
            img = np.expand_dims(img, axis=0)  # [H,W] -> [C,H,W],C=1
            noisy_img = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32))
            noisy_img = noisy_img.unsqueeze(0).to(device)

            """" padding  """
            b, c, h, w = noisy_img.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            noisy_img = F.pad(noisy_img, (0, pad_w, 0, pad_h))

            """ denoise """
            noisy_img = (noisy_img / 255 - 0.5) * 2
            denoised_img, _ = model(noisy_img)
            denoised_img = 255 * (denoised_img + 1) / 2
            denoised_img = torch.floor(denoised_img + 0.5)  # 0.5 bias
            denoised_img = denoised_img.clamp(0, 255)

            """ inverse padding """
            denoised_img = denoised_img[:, :, :h, :w]

            """ save results """
            denoised_img = denoised_img.squeeze(0).cpu()
            denoised_img = denoised_img.permute(1, 2, 0).numpy()

            cv2.imwrite(os.path.join(args.output_dir, img_path[:-4] + '_DN.png'), denoised_img)

        print('Test the model: ', f'{args.model_weight_dir} is Done! Time used is ', time.time() - time_begin)






