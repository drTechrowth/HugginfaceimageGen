import os
import io
from PIL import Image
import base64
import requests
import gradio as gr

HF_TOKEN = os.environ["HF_TOKEN"]
HF_API_TTI_BASE = os.environ["HF_API_TTI_BASE"]  # Your image endpoint

# Use the Llama 3.1-8B-Instruct in single-prompt, single-response mode
def rewrite_prompt(user_intent):
    url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    # Single prompt, single response—no chat format
    data = {
        "inputs": f"Rewrite as a detailed, vivid image prompt for a text-to-image AI: {user_intent}"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise RuntimeError(f"Prompt rewrite failed: {response.status_code} {response.text}")
    result = response.json()
    # Response is typically a list with dicts containing 'generated_text'
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"].strip()
    else:
        raise RuntimeError(f"Unexpected prompt rewrite response: {result}")

def get_completion(enhanced_prompt, parameters=None, ENDPOINT_URL=HF_API_TTI_BASE):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"inputs": enhanced_prompt}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
    try:
        result = response.json()
        if isinstance(result, dict) and "images" in result:
            return result["images"][0]
        if isinstance(result, dict) and "data" in result:
            return result["data"][0]
        if isinstance(result, list):
            return result[0]
        if isinstance(result, str):
            return result
        raise RuntimeError("Unknown response format from Hugging Face API.")
    except Exception:
        if response.status_code == 200:
            img_bytes = response.content
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            return img_base64
        else:
            raise RuntimeError(f"API returned status code {response.status_code}: {response.text}")

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(user_intent, negative_prompt, steps, guidance, width, height):
    # Step 1: Rewrite the prompt (user never sees this)
    enhanced_prompt = rewrite_prompt(user_intent)
    # Step 2: Use the enhanced prompt to generate the image
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(enhanced_prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation")
    with gr.Row():
        with gr.Column(scale=4):
            user_intent = gr.Textbox(label="Describe your idea or intention (natural language)")
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Generate")
    with gr.Accordion("Advanced options", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt")
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25)
                guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7)
            with gr.Column():
                width = gr.Slider(label="Width", minimum=64, maximum=1024, step=64, value=768)
                height = gr.Slider(label="Height", minimum=64, maximum=1024, step=64, value=768)
    output = gr.Image(label="Result")
    btn.click(
        fn=generate,
        inputs=[user_intent, negative_prompt, steps, guidance, width, height],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
