import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils import data
from advent.model.transoformer import SwinTransformer
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet
from advent.utils.func import prob_2_entropy
import torch.nn.functional as F



img_file = ''
input_size = 512
interpolation = Image.BICUBIC
NUM_CLASSES = 2
MULTI_LEVEL = True
device = 0
# pth location
restore_from = ''

palette = [0, 0, 0, 255, 172, 0]
def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)#,map_location={'cuda:1':'cuda:0'})
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    img = Image.open(img_file)
    img = img.resize((input_size, input_size), interpolation)
    img_input = np.asarray(img, np.float32)[:, :, ::-1].transpose((2, 0, 1))
    img_input = img_input[None].copy()
    img_input = torch.from_numpy(img_input)
    model_gen = SwinTransformer()
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    interp_target = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
    _, pred_trg_main = model_gen(img_input.cuda(device))
    pred_trg_main = interp_target(pred_trg_main)
    output_np_tensor = pred_trg_main.cpu().data[0].numpy()
    mask_np_tensor = output_np_tensor.transpose(1, 2, 0)
    mask_np_tensor = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_color = colorize(mask_np_tensor)
    img_name = img_file.split('/')[-1].split('.')[0]
    mask_color.save('./Result/%s_uda.png' % img_name)
    print('end')

if __name__ == '__main__':
    main()
