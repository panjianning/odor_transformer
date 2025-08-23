# DeepChem优化指南

本文档介绍如何使用DeepChem库优化分子气味预测项目，并进行Transformer模型与图神经网络的对比实验。

## 已实现的优化

### 1. 图神经网络基准模型
- **GraphConvModel**: 经典的图卷积网络
- **MPNNModel**: 消息传递神经网络  
- **AttentiveFPModel**: 带注意力机制的图网络

### 2. Transformer模型优化
- **分子特征增强**: 使用DeepChem的分子指纹特征
- **多模态融合**: 结合序列表示和结构特征
- **注意力机制改进**: 增强的注意力池化层

### 3. 评估指标优化
- **Macro AUC**: 多标签分类的宏观AUC
- **Macro F1**: 多标签分类的宏观F1分数
- **准确率**: 整体分类准确率

## 文件结构

```
models/
├── graph_models.py          # 图神经网络实现
├── optimized_transformer.py # 优化的Transformer模型
├── property_model.py        # 原始性质预测模型
└── transformer_model.py     # 基础Transformer模型

experiments/
└── model_comparator.py      # 模型对比实验框架

run_odor_comparison.py       # 主对比实验脚本
run_comparison.py            # 完整对比实验脚本
```

## 使用方法

### 1. 运行对比实验

```bash
# 使用示例数据运行实验
python run_odor_comparison.py --data sample_training_data.csv --output ./results

# 使用完整数据运行实验  
python run_odor_comparison.py --data ../input/smiles.csv --output ./full_results
```

### 2. 单独测试模型

```python
from models.graph_models import GraphModelWrapper

# 初始化图模型
graph_model = GraphModelWrapper(
    model_type='GraphConv',
    n_tasks=138,
    mode='classification'
)

# 训练模型
graph_model.train(train_smiles, train_labels, epochs=100)

# 预测
predictions, indices = graph_model.predict(test_smiles)
```

### 3. 使用优化的Transformer

```python
from models.optimized_transformer import EnhancedTransformerModel

# 创建优化模型
model = EnhancedTransformerModel(
    num_tokens=306,
    d_model=256,
    num_heads=8,
    d_hidden=1024,
    num_layers=8,
    num_labels=138,
    use_fingerprint=True
)

# 使用分子特征增强
output = model(smiles_ids, smiles_list)
```

## 性能指标

### 主要评估指标
1. **Macro AUC**: 所有标签的AUC平均值
2. **Macro F1**: 所有标签的F1分数平均值  
3. **Accuracy**: 整体分类准确率

### 基线模型
- 随机猜测 (Random)
- 多数类 (Majority) 
- 频率基线 (Frequency)

## DeepChem功能利用

### 分子特征提取
```python
from deepchem.feat import CircularFingerprint, Mol2VecFeaturizer

# 分子指纹特征
fingerprint = CircularFingerprint(size=2048)
features = fingerprint.featurize(smiles_list)

# Mol2Vec特征
mol2vec = Mol2VecFeaturizer()
features = mol2vec.featurize(smiles_list)
```

### 图神经网络模型
```python
from deepchem.models import GraphConvModel, MPNNModel, AttentiveFPModel

# 图卷积网络
model = GraphConvModel(n_tasks=138, mode='classification')

# 消息传递网络  
model = MPNNModel(n_tasks=138, mode='classification')

# 注意力FP网络
model = AttentiveFPModel(n_tasks=138, mode='classification')
```

### 数据管理
```python
from deepchem.data import NumpyDataset

# 创建数据集
dataset = NumpyDataset(features, labels)

# 模型训练
model.fit(dataset, nb_epoch=100)
```

## 优化建议

### 1. 数据层面
- 使用DeepChem的数据增强功能
- 利用分子描述符作为额外特征
- 实现SMILES扩充技术

### 2. 模型层面  
- 尝试不同的图神经网络架构
- 实验多模态融合策略
- 使用预训练的分子表示

### 3. 训练层面
- 实现课程学习策略
- 使用类别平衡损失函数
- 优化超参数搜索

## 结果分析

运行对比实验后，查看生成的JSON结果文件，包含：

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "dataset_info": {
    "n_samples": 1000,
    "n_labels": 138,
    "positive_ratio": 0.15
  },
  "results": {
    "random": {"macro_auc": 0.5, "macro_f1": 0.1, "accuracy": 0.85},
    "transformer": {"macro_auc": 0.85, "macro_f1": 0.75, "accuracy": 0.92},
    "graphconv": {"macro_auc": 0.82, "macro_f1": 0.72, "accuracy": 0.90}
  }
}
```

## 后续工作

1. **扩展数据集**: 使用更大的气味数据集
2. **模型融合**: 集成多个模型的预测结果
3. **超参数优化**: 使用DeepChem的超参数搜索工具
4. **可解释性**: 添加模型可解释性分析
5. **部署优化**: 优化模型推理速度

通过DeepChem的丰富功能，可以显著提升分子气味预测模型的性能和可扩展性。
