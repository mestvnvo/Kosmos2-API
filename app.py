import gradio as gr
from transformers import pipeline

# pipeline as high level
pipe = pipeline("image-text-to-text", model="HuggingFaceM4/Idefics3-8B-Llama3")

SYSTEM_PROMPT = f"""
You are a image vibe AI and your job is to help users capture the energy 
and aesthetic of the entire scene - mainly what the people are doing in their background.
Respond only in one sentence and don't elaborate.
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
                {"type": "image", "image": image}
            ]
        }
    ]

    result = pipe(messages)
    return result[0]['generated_text']

# api w/ gradio
api = gr.Interface(
    fn=get_image_vibe,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs="text"
)

api.launch(show_api=True)
