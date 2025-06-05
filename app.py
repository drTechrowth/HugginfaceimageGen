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
