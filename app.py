import os
import io
import base64
from PIL import Image
import requests
import gradio as gr
from dotenv import load_dotenv

# Load env vars if running locally
load_dotenv()

HF_API_KEY = os.environ["HF_API_KEY"]
HF_API_TTI_BASE = os.environ["HF_API_TTI_BASE"]  # e.g. https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1

# Pick a fast/reliable text-to-text LLM endpoint. You may swap to another supported model.
LLM_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def rewrite_prompt(user_intent: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"Rewrite as a vivid, detailed, high-quality image prompt for a text-to-image AI: {user_intent.strip()}"
    }
    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"].strip()
        else:
            raise RuntimeError(f"Unexpected LLM API response: {result}")
    except Exception as e:
        raise RuntimeError(f"Prompt rewrite failed: {e}")

def get_image_from_prompt(prompt: str, params: dict) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": params
    }
    try:
        response = requests.post(HF_API_TTI_BASE, headers=headers, json=payload, timeout=120)
        # Try to parse as JSON
        try:
            result = response.json()
            if isinstance(result, dict) and "images" in result:
                return result["images"][0]
            elif isinstance(result, dict) and "data" in result:
                return result["data"][0]
            elif isinstance(result, list):
                return result[0]
            elif isinstance(result, str):
                return result
            else:
                raise RuntimeError("Unknown response format from image API.")
        except Exception:
            # If not JSON, assume it's raw image bytes
            if response.status_code == 200:
                img_bytes = response.content
                return base64.b64encode(img_bytes).decode("utf-8")
            else:
                raise RuntimeError(f"Image API returned status code {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")

def base64_to_pil(img_base64: str) -> Image.Image:
    decoded = base64.b64decode(img_base64)
    return Image.open(io.BytesIO(decoded))

def generate(user_intent, negative_prompt, steps, guidance, width, height):
    # 1. Rewrite user intent as a strong image prompt
    enhanced_prompt = rewrite_prompt(user_intent)
    # 2. Generate the image
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    img_base64 = get_image_from_prompt(enhanced_prompt, params)
    pil_img = base64_to_pil(img_base64)
    return pil_img

with gr.Blocks() as demo:
    gr.Markdown("# 🔮 Text-to-Image Generator")
    with gr.Row():
        user_intent = gr.Textbox(label="Describe your idea (natural language)", placeholder="A cat riding a bicycle in a magical world")
        btn = gr.Button("Generate Image")
    with gr.Accordion("Advanced options", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt (optional)", placeholder="blurry, low quality, watermark")
        with gr.Row():
            steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25)
            guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7)
            width = gr.Slider(label="Width", minimum=64, maximum=1024, step=64, value=768)
            height = gr.Slider(label="Height", minimum=64, maximum=1024, step=64, value=768)
    output = gr.Image(label="Generated Image")
    btn.click(
        fn=generate,
        inputs=[user_intent, negative_prompt, steps, guidance, width, height],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
