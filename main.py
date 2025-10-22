import argparse
import transformers
import diffusers
from src.config import Config
from src.data import preprocess_images
from src.training import train_lora
from src.web import create_demo
from src.utils.logger import get_logger
 
# Suppress excessive HF logging
transformers.logging.set_verbosity_error()
diffusers.utils.logging.set_verbosity_error()

logger = get_logger("art_lora.main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing, training, or demo for LoRA fine-tuning."
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "demo"],
        required=True,
        help="Choose which stage to run: preprocess | train | demo"
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="q1w2e3_person",
        help="Unique token used to identify the trained concept (e.g., your_person_name_person)"
    )

    args = parser.parse_args()
    config = Config()

    if args.stage == "preprocess":
        logger.info("Starting preprocessing...")
        preprocess_images("data/original_photos", "data/processed_photos", target_size=config.resolution)
        logger.info("Preprocessing completed successfully.")

    elif args.stage == "train":
        logger.info("Starting LoRA fine-tuning...")
        train_lora(config)
        logger.info("Training completed successfully.")

    elif args.stage == "demo":
        logger.info("Launching demo interface...")
        create_demo(
            config.pretrained_model_name,
            f"{config.output_dir}/pytorch_lora_weights.safetensors",
            args.prompt_name
        )
        logger.info("Demo launched successfully! Open the Gradio app in your browser.")
