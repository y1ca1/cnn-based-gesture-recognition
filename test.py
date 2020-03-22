from PIL import Image
import torchvision.transforms as T
from torch.autograd import Variable as V
import torch
import os
from gest_model import ThreeLayerConvNet, TwoLayerConvNet
from our_parser import get_config
config = get_config()

trans = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225])
])

model = TwoLayerConvNet(config)  # 导入网络模型
model.eval()
# load trained model
model_dir = '%s/TwoLayerConv/model.pkl' % config['result_dir']
if os.path.exists(model_dir):
    model.load_state_dict(torch.load(model_dir))# 加载训练好的模型文件
    print('Model is loaded from %s' % model_dir)
else:
    print('[Error] cannot find model file.')


# 读入图片
def off_line():
    store_dir = '%s/%s' % (config['root_dir'], 'test_pics_yi')
    for root, dirs, files in os.walk(store_dir, topdown=True):
        for pics in files:
            print('Groundtruth Label:', pics[0])
            img = Image.open(os.path.join(store_dir, pics))
            # img.resize((64, 64))
            img = T.functional.resize(img, 64)
            img = T.functional.rotate(img, -90)
            # print(img.size)
            img = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
            img = img.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
            img_input = V(img)
            scores = model(img_input)  # 将图片输入网络得到输出
            _, preds = scores.max(1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
            print('Predicted Label:', int(preds.numpy()))


import cv2
import time


def real_time():
    capture = cv2.VideoCapture(0)
    while True:
        # 获取一帧
        cv2.namedWindow('camera')
        ret, frame = capture.read()
        cv2.imshow('camera', frame)
        cv2.waitKey(1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # img = T.functional.resize(image, (64, 64))
        # img = T.functional.rotate(img, -90)
        # print(img.size)
        img = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
        img = img.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
        img_input = V(img)
        scores = model(img_input)  # 将图片输入网络得到输出
        _, preds = scores.max(1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
        print('Predicted Label:', int(preds.numpy()))
        time.sleep(0.1)

# real_time()
off_line()