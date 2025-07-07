"""
快速训练验证脚本
用于快速验证训练效果，训练次数很少，便于调试
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import time

from config import Config
from data_processor import DataProcessor
from models.LSTM import LSTMModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_train_validation():
    """快速训练验证"""
    try:
        logger.info("开始快速训练验证...")
        
        # 加载配置
        config = Config()
        logger.info(f"配置: 序列长度={config.SEQUENCE_LENGTH}, 预测长度={config.PREDICTION_LENGTH}, "
                   f"批次大小={config.BATCH_SIZE}, 训练轮数={config.NUM_EPOCHS}")
        
        # 初始化数据处理器
        processor = DataProcessor(config)
        
        # 加载数据
        df = processor.load_data()
        logger.info(f"数据加载成功，形状: {df.shape}")
        
        # 准备特征
        features, labels = processor.prepare_features(df)
        logger.info(f"特征准备完成，特征形状: {features.shape}, 标签形状: {labels.shape}")
        
        # 创建序列数据（只使用前1000个样本进行快速验证）
        X, y = processor.create_sequences(features, labels)
        logger.info(f"序列数据创建完成，X形状: {X.shape}, y形状: {y.shape}")
        
        # 只使用前1000个样本
        X_quick = X[:1000]
        y_quick = y[:1000]
        logger.info(f"使用快速验证样本，X形状: {X_quick.shape}, y形状: {y_quick.shape}")
        
        # 数据分割
        train_size = int(0.8 * len(X_quick))
        val_size = len(X_quick) - train_size
        
        X_train = X_quick[:train_size]
        y_train = y_quick[:train_size]
        X_val = X_quick[train_size:]
        y_val = y_quick[train_size:]
        
        logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        # 初始化模型
        feature_dim = X_train.shape[2]
        model = LSTMModel(
            input_dim=feature_dim,
            hidden_size=64,  # 使用更小的隐藏层
            num_layers=2,
            prediction_length=config.PREDICTION_LENGTH,
            dropout=0.2,
            use_layernorm=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"模型初始化完成，使用设备: {device}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练设置
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # 训练循环
        logger.info("开始训练...")
        start_time = time.time()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(config.NUM_EPOCHS):
            # 训练
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                       f"Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        training_time = time.time() - start_time
        logger.info(f"训练完成！总用时: {training_time:.2f}秒")
        
        # 保存模型
        model_path = Path("checkpoints/quick_train_model.pth")
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存到: {model_path}")
        
        # 简单预测测试
        logger.info("进行预测测试...")
        model.eval()
        with torch.no_grad():
            test_X = torch.FloatTensor(X_val[:5]).to(device)  # 测试前5个样本
            predictions = model(test_X)
            logger.info(f"预测形状: {predictions.shape}")
            logger.info(f"预测值范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
        # 保存训练曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='训练损失', marker='o')
            plt.plot(val_losses, label='验证损失', marker='s')
            plt.title('Quick Training Validation - Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plot_path = Path("plots/quick_train_loss.png")
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"训练曲线已保存到: {plot_path}")
        except Exception as e:
            logger.warning(f"绘制训练曲线失败: {str(e)}")
        
        # 生成验证报告
        report_path = Path("quick_train_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("快速训练验证报告\n")
            f.write("="*30 + "\n\n")
            f.write(f"训练时间: {training_time:.2f}秒\n")
            f.write(f"训练轮数: {config.NUM_EPOCHS}\n")
            f.write(f"最终训练损失: {train_losses[-1]:.6f}\n")
            f.write(f"最终验证损失: {val_losses[-1]:.6f}\n")
            f.write(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"使用设备: {device}\n")
            f.write(f"样本数量: {len(X_quick)}\n")
            f.write(f"批次大小: {config.BATCH_SIZE}\n")
            
            if len(val_losses) > 1:
                loss_improvement = val_losses[0] - val_losses[-1]
                f.write(f"验证损失改善: {loss_improvement:.6f}\n")
        
        logger.info(f"验证报告已保存到: {report_path}")
        
        # 判断训练是否成功
        if val_losses[-1] < val_losses[0]:  # 验证损失下降
            logger.info("✅ 快速训练验证成功！模型正在学习。")
            logger.info("💡 提示: 现在可以运行完整训练或对比实验。")
            return True
        else:
            logger.warning("⚠️ 验证损失没有下降，可能需要调整参数。")
            return False
            
    except Exception as e:
        logger.error(f"快速训练验证失败: {str(e)}")
        return False

def check_training_requirements():
    """检查训练要求"""
    logger.info("检查训练要求...")
    
    # 检查数据文件
    config = Config()
    data_path = Path(config.DATA_PATH)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        return False
    
    # 检查数据大小
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        logger.info(f"数据文件大小: {len(df)} 行")
        if len(df) < 1000:
            logger.warning("数据量较少，可能影响训练效果")
    except Exception as e:
        logger.error(f"读取数据文件失败: {str(e)}")
        return False
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("使用CPU训练")
    
    # 检查必要目录
    for dir_path in ['checkpoints', 'plots', 'logs']:
        Path(dir_path).mkdir(exist_ok=True)
    
    logger.info("✅ 训练要求检查通过")
    return True

if __name__ == "__main__":
    # 检查训练要求
    if check_training_requirements():
        # 运行快速训练验证
        success = quick_train_validation()
        if success:
            print("\n🎉 快速训练验证成功！")
            print("现在您可以:")
            print("1. 运行完整训练: python train.py")
            print("2. 运行模型对比: python compare_models_enhanced.py")
            print("3. 进行预测: python predict.py")
        else:
            print("\n❌ 快速训练验证失败，请检查配置和数据。")
    else:
        print("\n❌ 训练要求检查失败，请解决问题后重试。") 