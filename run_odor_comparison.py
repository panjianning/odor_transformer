"""气味预测模型对比实验

专注于macro AUC和macro F1指标的多标签分类对比实验
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, f1_score
from csv_processor import CSVProcessor

class OdorModelComparator:
    """气味预测模型对比器
    
    专注于macro AUC和macro F1指标的对比实验
    """
    
    def __init__(self, data_path: str, output_dir: str = './odor_results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.csv_processor = CSVProcessor()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.smiles_list, self.labels, self.label_to_id, self.label_names = self._load_data()
        self.num_labels = len(self.label_names)
        
        print(f"数据集信息:")
        print(f"  样本数量: {len(self.smiles_list)}")
        print(f"  标签数量: {self.num_labels}")
        print(f"  正样本比例: {np.mean(self.labels):.3f}")
    
    def _load_data(self):
        """加载数据"""
        return self.csv_processor.load_multi_label_csv(self.data_path)
    
    def _split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """划分数据集"""
        n_samples = len(self.smiles_list)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return (train_indices, val_indices, test_indices)
    
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实标签 [n_samples, n_labels]
            y_pred: 预测概率 [n_samples, n_labels]
            
        Returns:
            Dict[str, float]: 评估指标
        """
        metrics = {}
        
        try:
            # Macro AUC
            metrics['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
        except:
            metrics['macro_auc'] = 0.0
        
        try:
            # Macro F1 (需要二值化预测)
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['macro_f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        except:
            metrics['macro_f1'] = 0.0
        
        # 其他指标
        metrics['accuracy'] = np.mean(y_pred_binary == y_true)
        
        return metrics
    
    def run_baseline_experiments(self):
        """运行基线实验"""
        print("\n" + "="*60)
        print("运行基线实验")
        print("="*60)
        
        results = {}
        
        # 随机猜测基线
        print("1. 随机猜测基线...")
        random_preds = np.random.rand(len(self.smiles_list), self.num_labels)
        results['random'] = self.evaluate_metrics(np.array(self.labels), random_preds)
        
        # 多数类基线
        print("2. 多数类基线...")
        majority_preds = np.zeros((len(self.smiles_list), self.num_labels))
        results['majority'] = self.evaluate_metrics(np.array(self.labels), majority_preds)
        
        # 频率基线（每个标签独立预测）
        print("3. 频率基线...")
        label_freq = np.mean(self.labels, axis=0)
        freq_preds = np.tile(label_freq, (len(self.smiles_list), 1))
        results['frequency'] = self.evaluate_metrics(np.array(self.labels), freq_preds)
        
        return results
    
    def run_transformer_experiment(self):
        """运行Transformer模型实验"""
        print("\n4. Transformer模型实验...")
        
        try:
            from training.property_trainer import PropertyTrainer, PropertyPredConfig
            
            # 创建临时数据文件
            temp_data = pd.DataFrame({
                'token_ids': [str(i) for i in range(len(self.smiles_list))],
                'label_ids': [str(label) for label in self.labels]
            })
            temp_path = os.path.join(self.output_dir, 'temp_data.csv')
            temp_data.to_csv(temp_path, index=False)
            
            # 配置和训练
            config = PropertyPredConfig(
                num_tokens=306,
                num_labels=self.num_labels,
                dim_embed=128,
                dim_tf_hidden=512,
                num_head=8,
                num_layers=6,
                dropout=0.1,
                num_epoch=50  # 减少epochs用于演示
            )
            
            trainer = PropertyTrainer(config, os.path.join(self.output_dir, 'transformer'))
            test_metrics = trainer.train(temp_path)
            
            # 清理
            os.remove(temp_path)
            
            return {
                'macro_auc': test_metrics.get('auc_macro', 0),
                'macro_f1': test_metrics.get('f1_macro', 0),
                'accuracy': test_metrics.get('accuracy', 0)
            }
            
        except Exception as e:
            print(f"Transformer实验失败: {e}")
            return {'macro_auc': 0, 'macro_f1': 0, 'accuracy': 0}
    
    def run_graph_model_experiment(self, model_type: str):
        """运行图模型实验"""
        print(f"5. {model_type} 图模型实验...")
        
        try:
            from models.graph_models import GraphModelWrapper
            
            # 划分数据
            train_idx, val_idx, test_idx = self._split_data()
            
            train_smiles = [self.smiles_list[i] for i in train_idx]
            train_labels = [self.labels[i] for i in train_idx]
            test_smiles = [self.smiles_list[i] for i in test_idx]
            test_labels = [self.labels[i] for i in test_idx]
            
            # 训练图模型
            graph_model = GraphModelWrapper(
                model_type=model_type,
                n_tasks=self.num_labels,
                mode='classification',
                batch_size=16,
                learning_rate=0.001
            )
            
            graph_model.train(train_smiles, train_labels, epochs=30)
            
            # 预测和评估
            test_preds, valid_idx = graph_model.predict(test_smiles)
            test_labels = np.array(test_labels)[valid_idx]
            
            return self.evaluate_metrics(test_labels, test_preds)
            
        except Exception as e:
            print(f"{model_type}实验失败: {e}")
            return {'macro_auc': 0, 'macro_f1': 0, 'accuracy': 0}
    
    def run_all_experiments(self):
        """运行所有实验"""
        all_results = {}
        
        # 基线实验
        all_results.update(self.run_baseline_experiments())
        
        # Transformer实验
        all_results['transformer'] = self.run_transformer_experiment()
        
        # 图模型实验
        graph_models = ['GraphConv', 'MPNN', 'AttentiveFP']
        for model_type in graph_models:
            all_results[model_type.lower()] = self.run_graph_model_experiment(model_type)
        
        # 保存结果
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict):
        """保存结果"""
        import json
        from datetime import datetime
        
        result_file = os.path.join(self.output_dir, f'odor_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'n_samples': len(self.smiles_list),
                    'n_labels': self.num_labels,
                    'positive_ratio': np.mean(self.labels)
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {result_file}")
        self._print_results(results)
    
    def _print_results(self, results: Dict):
        """打印结果"""
        print("\n" + "="*60)
        print("实验结果汇总 (Macro指标)")
        print("="*60)
        print(f"{'模型':<15} {'Macro AUC':<10} {'Macro F1':<10} {'Accuracy':<10}")
        print("-" * 50)
        
        for model_name, metrics in results.items():
            auc = metrics.get('macro_auc', 0)
            f1 = metrics.get('macro_f1', 0)
            acc = metrics.get('accuracy', 0)
            print(f"{model_name:<15} {auc:.4f}     {f1:.4f}     {acc:.4f}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='气味预测模型对比实验')
    parser.add_argument('--data', type=str, default='sample_training_data.csv', help='数据文件路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    
    args = parser.parse_args()
    
    print("气味预测模型对比实验")
    print("专注于Macro AUC和Macro F1指标")
    print("="*60)
    
    comparator = OdorModelComparator(args.data, args.output)
    results = comparator.run_all_experiments()
    
    print("\n实验完成!")

if __name__ == "__main__":
    main()
