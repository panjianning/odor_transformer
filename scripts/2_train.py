from training.property_trainer import PropertyTrainer
from models.property_model import PropertyPredConfig

if __name__ == '__main__':
    # 创建训练器
    trainer = PropertyTrainer(
        config=PropertyPredConfig(),
        save_dir="output/models/",
        device="cpu"
    )

    # 开始训练
    test_metrics = trainer.train("output/data.csv",resume_from_checkpoint=False)
    print(test_metrics)