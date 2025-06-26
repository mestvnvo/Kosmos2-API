import gradio as gr
from transformers import pipeline

# pipeline as high level
pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-500M-Instruct")

SYSTEM_PROMPT = f"""
You are a image vibe AI and your job is to help users capture the energy and aesthetic of the scene.
"""

def get_image_vibe(image):
    if image is None:
        return "No image provided."

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What are the people doing?"}
            ]
        }
    ]

    result = pipe(messages)
    return result[0]['generated_text'][2]["content"]

# api w/ gradio
api = gr.Interface(
    fn=get_image_vibe,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs="text"
)

api.launch(show_api=True)
