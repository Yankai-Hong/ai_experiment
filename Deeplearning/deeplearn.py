import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch import device
from torchvision.transforms import transforms
from model import *
from mydataset import *
import time

device = device("cuda" if torch.cuda.is_available() else "cpu")


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()])
# 准备数据集

train_data = torchvision.datasets.ImageFolder('pic/train', transform=transform)
test_data = MyDataset('pic/test', transform=transform)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# dataloader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
net = Net().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 1e-5
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

# 设置训练网络参数:
# --------------
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练次数
epoch = 30

start_time = time.time()

# 添加tensorboard
writer = SummaryWriter('./logs_train_last')

# early stopping
counter = 0
min_loss = None

for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")

    # 训练步骤开始
    net.train()
    total_train_loss = 0
    total_train_accuracy = 0

    for data in train_dataloader:
        # forward
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = net(imgs)
        end_time = time.time()

        # 损失函数
        loss = loss_fn(outputs, targets)
        # loss = nn.MSMSELoss(outputs, targets)

        # 预测正确率
        accuracy = (outputs.argmax(1) == targets).sum()
        total_train_accuracy += accuracy.item()

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        # print(end_time - start_time)

    total_train_step += 1
    train_accuracy_percent = total_train_accuracy / train_data_size
    print(f'epoch[{i + 1}/{epoch}], train_loss = {total_train_loss}, train_accuracy = {train_accuracy_percent}')
    writer.add_scalar('train_loss', loss.item(), total_train_step)
    writer.add_scalar('train_accuracy', train_accuracy_percent, total_train_step)

    if i == epoch - 1:
        torch.save(net, 'net_method_last.pth')
        print('模型已保存')

    if min_loss is None:
        min_loss = total_train_loss
    elif total_train_loss > min_loss:
        counter += 1

        if counter > 3:
            print("Early Stopping")
            break
    elif total_train_loss < min_loss:
        counter = 0
        min_loss = total_train_loss

writer.close()

# 测试数据集
total_correct = 0
total_samples = 0

with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        total_correct = (outputs.argmax(1) == targets).sum()
        total_samples += targets.size(0)

print(f"test accuracy = {total_correct / total_samples}")
