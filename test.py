from PIL import Image
import torchvision.transforms as T
from models.mobilenet_master2 import MobileNet  # 导入自己定义的网络模型
from torch.autograd import Variable as V
import torch as t

trans = T.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 读入图片
img = Image.open('Z:\\毕业\\1\\0_0.JPG')
in_put = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
img = img.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]

model = MobileNet().cuda()  # 导入网络模型
model.eval()
model.load_state_dict(t.load('Z:\\毕业\\已经训练好的Cityscapes模型？\\OCNet.pytorch-master\\run_resnet101_asp_oc.sh'))  # 加载训练好的模型文件

in_put = V(img.cuda())
score = model(input)  # 将图片输入网络得到输出
probability = t.nn.functional.softmax(score, dim=1) # 计算softmax，即该图片属于各类的概率
max_value, index = t.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
print(index)
