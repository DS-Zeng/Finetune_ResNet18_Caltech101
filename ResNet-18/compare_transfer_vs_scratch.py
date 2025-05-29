import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import copy
import os

# 数据集路径和超参数
data_dir = 'data/caltech-101/101_ObjectCategories'
batch_size = 16
num_epochs = 50  # More epochs for fair comparison
learning_rate = 0.001

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) 
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) 
               for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"数据集加载完成，类别数量: {num_classes}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def create_model(use_pretrained=True):
    """创建模型，可选择是否使用预训练权重"""
    if use_pretrained:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print("使用预训练权重")
    else:
        model = models.resnet18(weights=None)
        print("随机初始化权重")
    
    # 替换最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

def load_pretrained_model(model_path):
    """加载已训练好的模型"""
    model = models.resnet18(weights=None)  # 不加载ImageNet权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 加载训练好的权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载预训练模型: {model_path}")
    else:
        print(f"警告: 模型文件 {model_path} 不存在!")
        return None
    
    return model.to(device)

def evaluate_model(model, dataloader, phase='test'):
    """评估模型性能"""
    model.eval()
    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating on {phase} set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # 计算每个类别的准确率
            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_acc = running_corrects.double() / total_samples
    overall_loss = running_loss / total_samples
    
    print(f'{phase.capitalize()} Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)')
    print(f'{phase.capitalize()} Loss: {overall_loss:.4f}')
    
    return overall_acc, overall_loss

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, experiment_name):
    """训练模型并记录到TensorBoard"""
    writer = SummaryWriter(f'runs/{experiment_name}')
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            with tqdm(dataloaders[phase], unit="batch", desc=f"{phase}") as tepoch:
                for inputs, labels in tepoch:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    tepoch.set_postfix(loss=loss.item())
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.cpu().numpy())
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    writer.close()
    
    return model, best_acc, time_elapsed

def main():
    """主函数：比较预训练模型和从零开始训练"""
    
    # 实验1：加载已训练好的Transfer Learning模型
    print("=" * 60)
    print("实验1：加载已训练好的Transfer Learning模型")
    print("=" * 60)
    
    pretrained_model_path = 'resnet18_caltech101_optimized.pth'
    model_pretrained = load_pretrained_model(pretrained_model_path)
    
    if model_pretrained is None:
        print("无法加载预训练模型，退出比较实验")
        return
    
    # 评估预训练模型在验证集和测试集上的性能
    print("\n评估预训练模型在验证集上的性能:")
    val_acc_pretrained, val_loss_pretrained = evaluate_model(model_pretrained, dataloaders['val'], 'validation')
    
    print("\n评估预训练模型在测试集上的性能:")
    test_acc_pretrained, test_loss_pretrained = evaluate_model(model_pretrained, dataloaders['test'], 'test')
    
    # 实验2：从零开始训练
    print("\n" + "=" * 60)
    print("实验2：Training from Scratch (随机初始化)")
    print("=" * 60)
    
    model_scratch = create_model(use_pretrained=False)
    criterion = nn.CrossEntropyLoss()
    
    # 为从零开始的模型使用统一学习率
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=learning_rate, momentum=0.9)
    scheduler_scratch = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=15, gamma=0.1)
    
    model_scratch, best_acc_scratch, time_scratch = train_model(
        model_scratch, dataloaders, criterion, optimizer_scratch, 
        scheduler_scratch, num_epochs, "training_from_scratch"
    )
    
    # 评估从零开始训练的模型在测试集上的性能
    print("\n评估从零训练模型在测试集上的性能:")
    test_acc_scratch, test_loss_scratch = evaluate_model(model_scratch, dataloaders['test'], 'test')
    
    # 保存从零开始训练的模型
    torch.save(model_scratch.state_dict(), '../resnet18_from_scratch.pth')
    
    # 比较结果
    print("\n" + "=" * 60)
    print("实验结果比较")
    print("=" * 60)
    print(f"Transfer Learning (已训练模型):")
    print(f"  - 验证集准确率: {val_acc_pretrained:.4f} ({val_acc_pretrained*100:.2f}%)")
    print(f"  - 测试集准确率: {test_acc_pretrained:.4f} ({test_acc_pretrained*100:.2f}%)")
    print(f"  - 验证集损失: {val_loss_pretrained:.4f}")
    print(f"  - 测试集损失: {test_loss_pretrained:.4f}")
    print(f"  - 训练时间: 已完成 (使用预训练模型)")
    
    print(f"\nTraining from Scratch:")
    print(f"  - 最佳验证准确率: {best_acc_scratch:.4f} ({best_acc_scratch*100:.2f}%)")
    print(f"  - 测试集准确率: {test_acc_scratch:.4f} ({test_acc_scratch*100:.2f}%)")
    print(f"  - 测试集损失: {test_loss_scratch:.4f}")
    print(f"  - 训练时间: {time_scratch/60:.1f} 分钟")
    print(f"  - 训练轮数: {num_epochs} epochs")
    
    print(f"\n性能对比:")
    val_acc_improvement = (val_acc_pretrained - best_acc_scratch) * 100
    test_acc_improvement = (test_acc_pretrained - test_acc_scratch) * 100
    print(f"  - 验证集准确率提升: {val_acc_improvement:.2f} 个百分点")
    print(f"  - 测试集准确率提升: {test_acc_improvement:.2f} 个百分点")
    print(f"  - Transfer Learning 在测试集上的优势: {test_acc_improvement:.2f}%")
    
    # 创建对比图表
    writer_comparison = SummaryWriter('runs/comparison')
    
    # 验证集准确率对比
    writer_comparison.add_scalars('Validation_Accuracy_Comparison', {
        'Transfer_Learning': val_acc_pretrained,
        'From_Scratch': best_acc_scratch
    }, 0)
    
    # 测试集准确率对比
    writer_comparison.add_scalars('Test_Accuracy_Comparison', {
        'Transfer_Learning': test_acc_pretrained,
        'From_Scratch': test_acc_scratch
    }, 0)
    
    # 测试集损失对比
    writer_comparison.add_scalars('Test_Loss_Comparison', {
        'Transfer_Learning': test_loss_pretrained,
        'From_Scratch': test_loss_scratch
    }, 0)
    
    # 训练时间对比 (假设Transfer Learning用了25个epoch，每个epoch约2分钟)
    estimated_transfer_time = 25 * 2  # 估计的训练时间（分钟）
    writer_comparison.add_scalars('Training_Time_Comparison', {
        'Transfer_Learning_Estimated': estimated_transfer_time,
        'From_Scratch': time_scratch/60
    }, 0)
    
    writer_comparison.close()
    
    print(f"\n可视化结果已保存到TensorBoard")
    print(f"运行 'tensorboard --logdir=runs' 查看详细比较")
    
    # 保存比较结果到文件
    results_file = 'comparison_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ResNet-18 Transfer Learning vs Training from Scratch 比较结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Transfer Learning (预训练模型):\n")
        f.write(f"  - 验证集准确率: {val_acc_pretrained:.4f} ({val_acc_pretrained*100:.2f}%)\n")
        f.write(f"  - 测试集准确率: {test_acc_pretrained:.4f} ({test_acc_pretrained*100:.2f}%)\n")
        f.write(f"  - 验证集损失: {val_loss_pretrained:.4f}\n")
        f.write(f"  - 测试集损失: {test_loss_pretrained:.4f}\n\n")
        f.write(f"Training from Scratch:\n")
        f.write(f"  - 最佳验证准确率: {best_acc_scratch:.4f} ({best_acc_scratch*100:.2f}%)\n")
        f.write(f"  - 测试集准确率: {test_acc_scratch:.4f} ({test_acc_scratch*100:.2f}%)\n")
        f.write(f"  - 测试集损失: {test_loss_scratch:.4f}\n")
        f.write(f"  - 训练时间: {time_scratch/60:.1f} 分钟\n\n")
        f.write(f"性能提升:\n")
        f.write(f"  - 验证集准确率提升: {val_acc_improvement:.2f} 个百分点\n")
        f.write(f"  - 测试集准确率提升: {test_acc_improvement:.2f} 个百分点\n")
    
    print(f"详细比较结果已保存到: {results_file}")

if __name__ == "__main__":
    main() 