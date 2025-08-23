"""测试数据处理和模型功能"""

import pandas as pd
from csv_processor import CSVProcessor

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载...")
    
    processor = CSVProcessor()
    
    try:
        # 加载数据
        smiles, labels, label_to_id, label_names = processor.load_multi_label_csv('sample_training_data.csv')
        
        print(f"成功加载 {len(smiles)} 个样本")
        print(f"标签数量: {len(label_names)}")
        print(f"标签维度: {len(labels[0]) if labels else 0}")
        
        # 显示一些样本信息
        print("\n前3个样本:")
        for i in range(min(3, len(smiles))):
            print(f"SMILES: {smiles[i]}")
            print(f"Labels: {labels[i]}")
            print(f"标签向量长度: {len(labels[i])}")
            print()
        
        # 统计标签分布
        label_counts = [0] * len(label_names)
        for label_vec in labels:
            for j, val in enumerate(label_vec):
                if val == 1:
                    label_counts[j] += 1
        
        print("标签分布统计:")
        for i, count in enumerate(label_counts):
            if count > 0:
                print(f"  {label_names[i]}: {count} 个样本")
        
        return True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建功能"""
    print("\n测试模型创建...")
    
    try:
        from models.property_model import PropertyPredConfig, PropertyPredictionModel
        
        # 创建模型配置
        config = PropertyPredConfig(
            num_tokens=306,
            num_labels=138,  # 138种气味
            dim_embed=128,
            dim_tf_hidden=512,
            num_head=8,
            num_layers=6,
            dropout=0.1
        )
        
        # 创建模型
        model = PropertyPredictionModel(config)
        
        print(f"模型创建成功")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试前向传播
        import torch
        test_input = torch.randint(0, 306, (2, 50))  # 批量大小2，序列长度50
        output = model(test_input)
        
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试数据处理和模型功能")
    print("=" * 60)
    
    # 测试数据加载
    data_ok = test_data_loading()
    
    # 测试模型创建
    model_ok = test_model_creation()
    
    print("=" * 60)
    if data_ok and model_ok:
        print("所有测试通过! ✓")
    else:
        print("测试失败! ✗")
    print("=" * 60)
