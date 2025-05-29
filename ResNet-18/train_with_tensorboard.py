import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import copy
import optuna

# 数据集路径和超参数
data_dir = 'data/caltech-101/101_ObjectCategories'
batch_size = 16  # Optimal batch size from hyperparameter search
num_epochs = 25
learning_rate = 0.00801  # Optimal learning rate for new layer
fine_tune_lr = 0.000957  # Optimal fine-tune learning rate

# 创建TensorBoard writer
writer = SummaryWriter('runs/resnet18_caltech101_experiment')

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

print("数据集加载完成……")
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"测试集大小: {dataset_sizes['test']}")
print(f"类别数量: {len(class_names)}")

# 加载预训练的 ResNet-18 模型
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 替换模型的最后一层以适应 Caltech-101
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备: {device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 为不同层组设置不同的学习率
other_params = [param for name, param in model.named_parameters() if "fc" not in name]
optimizer = optim.SGD([
    {'params': model.fc.parameters(), 'lr': learning_rate},
    {'params': other_params, 'lr': fine_tune_lr}
], momentum=0.9)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # 使用tqdm显示进度
            with tqdm(dataloaders[phase], unit="batch", desc=f"{phase}") as tepoch:
                for batch_idx, (inputs, labels) in enumerate(tepoch):
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
                    
                    # 更新进度条
                    tepoch.set_postfix(loss=loss.item())
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            # 记录学习率
            if phase == 'train':
                current_lr_fc = optimizer.param_groups[0]['lr']
                current_lr_features = optimizer.param_groups[1]['lr']
                writer.add_scalar('Learning_Rate/FC_Layer', current_lr_fc, epoch)
                writer.add_scalar('Learning_Rate/Feature_Layers', current_lr_features, epoch)
            
            # 保存历史记录
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.cpu().numpy())
            
            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 保存训练历史到TensorBoard
    for epoch in range(num_epochs):
        if epoch < len(train_losses):
            writer.add_scalars('Loss_Comparison', {
                'Train': train_losses[epoch],
                'Validation': val_losses[epoch]
            }, epoch)
            writer.add_scalars('Accuracy_Comparison', {
                'Train': train_accs[epoch],
                'Validation': val_accs[epoch]
            }, epoch)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_acc': best_acc
    }

def evaluate_model(model, dataloader, phase='test'):
    """评估模型在测试集上的性能"""
    model.eval()
    running_corrects = 0
    total_samples = 0
    
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating on {phase} set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # 计算每个类别的准确率
            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    overall_acc = running_corrects.double() / total_samples
    print(f'{phase.capitalize()} Accuracy: {overall_acc:.4f}')
    
    # 打印每个类别的准确率
    print("\nPer-class accuracy:")
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    return overall_acc


def objective(trial):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    fine_tune_lr = trial.suggest_loguniform('fine_tune_lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    # Update dataloders with the new batch size
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Define model and optimizer with new learning rates
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)

    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': learning_rate},
        {'params': [param for name, param in model.named_parameters() if "fc" not in name], 'lr': fine_tune_lr}
    ], momentum=0.9)

    model = model.to(device)

    # Train model
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)  # Use fewer epochs for faster search

    # Evaluate and return some metric (e.g., validation accuracy)
    val_acc = evaluate_model(trained_model, dataloaders['val'])
    return val_acc


if __name__ == "__main__":
    # Create a study and optimize the objective function
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)

    # print("Best hyperparameters:", study.best_params)
    # print("Best validation accuracy:", study.best_value)

    print("开始训练...")
    
    # 训练模型
    trained_model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), '../resnet18_caltech101_optimized.pth')
    print("模型已保存为 resnet18_caltech101_optimized.pth")
    
    # 在测试集上评估
    print("\n在测试集上评估模型性能:")
    test_acc = evaluate_model(trained_model, dataloaders['test'], 'test')
    
    # 记录最终测试准确率到TensorBoard
    writer.add_scalar('Final_Test_Accuracy', test_acc, 0)
    
    # 添加模型图到TensorBoard
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(trained_model, sample_input)
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f"\n实验完成!")
    print(f"最佳验证准确率: {history['best_acc']:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print("TensorBoard日志保存在 'runs/resnet18_caltech101_experiment' 目录")
    print("运行 'tensorboard --logdir=runs' 查看训练可视化") 