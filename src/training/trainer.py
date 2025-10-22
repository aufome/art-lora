# src/trainer.py

import os
import subprocess
import sys
import platform
import logging
from src.config import Config


logger = logging.getLogger(__name__)


def ensure_train_script():
    """Ensure train_dreambooth_lora.py exists. Download it if it is missing."""
    file_name = "train_dreambooth_lora.py"
    if not os.path.exists(file_name):
        logger.warning(f"{file_name} not found, downloading...")
        try:
            subprocess.run(
                [
                    "wget",
                    f"https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/{file_name}",
                ],
                check=True,
            )
            logger.info(f"{file_name} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {file_name}: {e}")
            raise
    else:
        logger.debug(f"{file_name} already exists.")


def ensure_peft():
    """Ensure PEFT is installed."""
    try:
        import peft  # noqa: F401
        logger.debug("PEFT already installed.")
    except ImportError:
        logger.warning("PEFT not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
            logger.info("PEFT installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PEFT: {e}")
            raise


def ensure_diffusers_source():
    """Ensure diffusers is up to date."""
    import diffusers
    from packaging import version

    if version.parse(diffusers.__version__) < version.parse("0.36.0.dev0"):
        logger.warning(
            f"Current diffusers version ({diffusers.__version__}) is outdated. Updating from source..."
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/diffusers"]
            )
            logger.info("Diffusers updated successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update diffusers: {e}")
            raise
    else:
        logger.debug("Diffusers version is up to date.")


def train_lora(config: Config):
    """Start LoRA fine-tuning."""
    logger.info("Preparing training environment...")

    # Remove unnecessary files
    for unwanted in [".DS_Store", ".gitkeep"]:
        path = os.path.join(config.instance_data_dir, unwanted)
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Removed {unwanted} from dataset directory.")

    # Ensure dependencies
    ensure_train_script()
    ensure_peft()
    ensure_diffusers_source()

    # Identify if the system has Apple Silicon.
    # Apple Silicon (M1/M2/M3/...) uses Metal instead of CUDA; disable 8-bit optimizer on these devices.
    system = platform.system()
    is_apple_silicon = system == "Darwin" and platform.machine() == "arm64"
    
    # Find the number processed photos
    num_images = len([f for f in os.listdir(config.instance_data_dir) 
                    if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')])

    
    logger.info(f"Number of training images found: {num_images}")

    num_epochs = 100

    # Set the maximum training steps dynamically
    MAX_TRAIN_STEPS = num_images * num_epochs
    logger.info(f"Maximum training steps set to: {MAX_TRAIN_STEPS}")

    # Build training command
    cmd = [
        "accelerate",
        "launch",
        "train_dreambooth_lora.py",
        f"--pretrained_model_name_or_path={config.pretrained_model_name}",
        f"--instance_data_dir={config.instance_data_dir}",
        f"--output_dir={config.output_dir}",
        f"--instance_prompt={config.instance_prompt}",
        f"--resolution={config.resolution}",
        f"--train_batch_size={config.train_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--checkpointing_steps={config.checkpointing_steps}",
        f"--learning_rate={config.learning_rate}",
        f"--lr_scheduler={config.lr_scheduler}",
        f"--lr_warmup_steps={config.lr_warmup_steps}",
        f"--max_train_steps={MAX_TRAIN_STEPS}",
        f"--mixed_precision={'no' if is_apple_silicon else config.mixed_precision}",
        f"--seed={config.seed}",
    ]

    # bitsandbytes (8-bit Adam optimizer) is only supported on CUDA-enabled systems
    if config.use_8bit_adam and not is_apple_silicon:
        cmd.append("--use_8bit_adam")
        logger.debug("8-bit Adam optimizer enabled.")
    else:
        logger.debug("8-bit Adam optimizer disabled (Apple Silicon or config disabled).")

    logger.info("Starting LoRA fine-tuning...")
    try:
        subprocess.run(cmd, check=True)
        logger.info("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training process failed: {e}")
        raise
