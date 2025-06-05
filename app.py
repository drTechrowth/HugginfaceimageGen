import os
import io
from PIL import Image
import base64
from dotenv import load_dotenv
import requests
import gradio as gr

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_TTI_BASE = os.getenv("HF_API_TTI_BASE").strip()  # Image
HF_API_LLM_BASE = os.getenv("HF_API_LLM_BASE").strip()  # Text-to-Text
PORT = int(os.getenv("PORT", 8080))

def call_hf_text2text(raw_prompt, ENDPOINT_URL=HF_API_LLM_BASE):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    # For instruct models, use a system prompt or instruction
    data = {
        "inputs": f"Rewrite the following user intention as a detailed, descriptive prompt for an AI image generator. Make it vivid and clear. User intention: '{raw_prompt}'"
    }
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise RuntimeError(f"LLM API returned status {response.status_code}: {response.text}")
    try:
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"LLM API Error: {result['error']}")
        # Output is typically a list of dicts with 'generated_text'
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        # Sometimes just a string
        if isinstance(result, str):
            return result.strip()
        raise RuntimeError("Unknown response format from LLM API.")
    except Exception as e:
        raise RuntimeError(f"Could not decode LLM JSON: {e}\nResponse content: {response.text}")

def get_completion(inputs, parameters=None, ENDPOINT_URL=HF_API_TTI_BASE):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
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
    # Step 1: Use LLM to rewrite the user's intent as a high-quality prompt
    enhanced_prompt = call_hf_text2text(user_intent)
    # Step 2: Use the enhanced prompt with the image API
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(enhanced_prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image, enhanced_prompt

with gr.Blocks() as demo:
    gr.Markdown("# Smart Image Generation")
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
    rewritten_prompt = gr.Textbox(label="Enhanced prompt used for image generation")
    btn.click(
        fn=generate,
        inputs=[user_intent, negative_prompt, steps, guidance, width, height],
        outputs=[output, rewritten_prompt]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
