from dataclasses import dataclass

@dataclass
class Config:
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    instance_data_dir: str = "data/processed_photos"
    output_dir: str = "model"
    instance_prompt: str = "q1w2e3_person"
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    checkpointing_steps: int = 250
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    mixed_precision: str = "fp16"
    seed: int = 42
    use_8bit_adam: bool = True
