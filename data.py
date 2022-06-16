from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def read_cifar10(batchsize,data_dir):
    # 数据变换
    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  # 随机旋转
                                    transforms.RandomCrop(32, padding=4),  # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  # 颜色变化。亮度
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),#Q1数据归一化问题：ToTensor是把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])#Q2均值方差问题：RGB3个通道分别进行正则化处理

    # 数据加载
    data_train = datasets.CIFAR10(root=data_dir,
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    data_test = datasets.CIFAR10(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 download=True
                                 )
    # 数据装载和数据预览
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,    #打乱数据
                                   pin_memory=False)   #内存充足时，可以设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False
                                   #drop_last=True)   #处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=False)
    return data_loader_train,data_loader_test
