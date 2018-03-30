'''
@author:sssste
https://github.com/sssste
'''

import os

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from PIL import ImageChops


def get_ImageNet_transform(resize_size=224, random_horizontal_flip=False):
    trans = []
    trans.append(transforms.Resize(resize_size))
    if random_horizontal_flip:
        trans.append(transforms.RandomHorizontalFlip())

    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(trans)


# 存储模块参数
def save_param(module, path):
    torch.save(module.state_dict(), path)
    print("save params to {} successful!".format(path))


def restore_param(module, path, strict=True):
    '''

    :param module: 载入的模块
    :param path: 权重存储的位置
    :param strict: 如果是，则严格按照所存储的权重载入，如果不是，只载入当前module所需要的权重（且存储文件中要有该权重）
    :return:
    '''
    pretrain_model_dict = torch.load(path)
    if not strict:
        pretrain_model_dict = {k: v for k, v in pretrain_model_dict.items() if k in module.model_dict()}
    module.load_state_dict(pretrain_model_dict)
    print("load params from {} successful!".format(path))


# 获得目前学习率
def get_learning_rate_from_optim(optim):
    return optim.param_groups[0]["lr"]


# 保存整个模型参数
def save_checkpoint(model_dict, global_step, now_batch, path):
    '''

    :param model_dict: 模型的参数，字典，key为某个保存的名字，value是模块
                        例：{
                            "generator_model":generator,
                            "discriminator_model":discriminator,
                            "optimizer":optimizer
                            }
                        最终会把generator的参数保存为"generator_model.pth"文件,以此类推

    :param global_step: 目前运行的梯度下降步数
    :param now_batch: 目前执行第几个batch
    :param path: 模型保存的地址
    :return: null
    '''
    if not path.endswith("/"):
        path = path + '/'

    for fname, module in model_dict.items():
        full_path = path + fname + ".pth"
        save_param(module, full_path)

    with open(path + "checkpoint", 'w') as f:
        f.writelines("global_step " + str(global_step) + '\n')
        f.writelines("now_batch " + str(now_batch) + '\n')


def restore_checkpoint(model_dict, path, strict=True):
    '''

    :param model_dict: 模型的参数，字典，key为某个保存的名字，value是模块
                        例：{
                            "generator_model":generator,
                            "discriminator_model":discriminator,
                            "optimizer":optimizer
                            }
                        最终会加载"generator_model.pth"的参数导generator模块中，以此类推
    :param path: 模型保存的地址
    :param strict: 如果是，则严格按照所存储的权重载入，如果不是，只载入当前module所需要的权重（且存储文件中要有该权重）
    :return: global_step,now_batch
    '''

    if not path.endswith("/"):
        path = path + '/'

    for fname, module in model_dict.items():
        full_path = path + fname + ".pth"
        if not os.path.exists(full_path):
            print(full_path + " not found!")
        else:
            restore_param(module, full_path, strict)

    if not os.path.exists(path + "checkpoint"):
        print("checkpoint not found")
        global_step = 0
        now_batch = 0
    else:
        with open(path + "checkpoint", "r")as f:
            fline = f.readlines()
            if len(fline) != 2:
                raise Exception("some params are missing")
            global_step = int(fline[0].split()[1])
            now_batch = int(fline[1].split()[1])

    return global_step, now_batch


def img_offset(img, xoff, yoff, loop=False):
    '''
    平移PIL图像

    :param img: 需要平移的PIL图片
    :param xoff: 图片横向平移的比例，范围在[-1,1]负数表示左移
    :param yoff: 图片纵向平移的比例，范围在[-1,1]负数表示上移
    :param loop: 平移是否循环，比如：左移的部分是否移动到右边
    :return:
    '''
    if xoff < -1 or xoff > 1 or yoff < -1 or yoff > 1:
        raise Exception("xoff or y off must in the range of [-1,1]")
    img = Image.fromarray(img)
    width, height = img.size
    xoff = int(xoff * width)
    yoff = int(yoff * height)
    c = ImageChops.offset(img, xoff, yoff)
    if not loop:
        if xoff >= 0:
            c.paste(0, box=(0, 0, xoff, height))
        else:
            c.paste(0, box=(width + xoff, 0, width, height))
        if yoff >= 0:
            c.paste(0, box=(0, 0, width, yoff))
        else:
            c.paste(0, box=(0, height + yoff, width, height))
    return c


def img_crop_scale(img, scale):
    if scale <= 0:
        raise Exception("scale must be in the range of (0，+inf)")

    old_size = img.size
    img = img.resize([int(old_size[0] * scale), int(old_size[1] * scale)])
    img = transforms.CenterCrop(old_size)(img)
    return img
