import os
import io
from PIL import Image
import base64
from dotenv import load_dotenv
import requests
import gradio as gr

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_TTI_BASE = os.getenv("HF_API_TTI_BASE").strip()  # Removes accidental whitespace
PORT = int(os.getenv("PORT", 8080))

def get_completion(inputs, parameters=None, ENDPOINT_URL=HF_API_TTI_BASE):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
    # Try to decode JSON first
    try:
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"API Error: {result['error']}")
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
        # If not JSON, treat as image bytes
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

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with FLUX.1-schnell (Hugging Face)")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt")
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit")
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
    btn.click(fn=generate, inputs=[prompt, negative_prompt, steps, guidance, width, height], outputs=[output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
