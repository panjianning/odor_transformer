"""简单测试脚本"""

from csv_processor import CSVProcessor

def main():
    print("测试数据加载...")
    
    processor = CSVProcessor()
    smiles, labels, label_to_id, label_names = processor.load_multi_label_csv('sample_training_data.csv')
    
    print(f"成功加载 {len(smiles)} 个样本")
    print(f"标签数量: {len(label_names)}")
    print(f"标签示例: {label_names[:5]}...")  # 显示前5个标签
    
    # 测试评估指标计算
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score
    
    # 创建一些测试数据
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.8, 0.2, 0.1], [0.1, 0.9, 0.2], [0.2, 0.1, 0.7]])
    
    # 计算macro指标
    macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred > 0.5, average='macro')
    
    print(f"\n测试指标计算:")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
