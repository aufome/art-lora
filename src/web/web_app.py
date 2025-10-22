import torch
import gradio as gr
import logging
from diffusers import StableDiffusionPipeline


logger = logging.getLogger(__name__)


def create_demo(model_id, lora_path, trigger_word):
    """Launch a Gradio demo for the trained LoRA model."""

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        logger.info("Using CUDA device for inference.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
        logger.info("Using Apple Silicon (MPS) device for inference.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.warning("No GPU detected. Running on CPU — this will be much slower.")

    try:
        logger.info(f"Loading Stable Diffusion pipeline from: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

        logger.info(f"Loading LoRA weights from: {lora_path}")
        pipe.load_lora_weights(lora_path)
    except Exception as e:
        logger.error(f"Failed to load model or LoRA weights: {e}")
        raise

    def generate(style):
        prompt = f"{trigger_word} {style}"
        logger.info(f"Generating image for prompt: '{prompt}'")
        try:
            image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            logger.info("Image generation completed successfully.")
            return image
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise

    logger.info("Launching Gradio interface...")
    demo = gr.Interface(fn=generate, inputs="text", outputs="image", title="LoRA Avatar Generator")
    demo.launch()
