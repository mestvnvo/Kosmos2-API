import gradio as gr
from transformers import pipeline
from PIL import Image

# pipeline as high level
pipe = pipeline("image-text-to-text", 
    model="microsoft/kosmos-2-patch14-224",
    device=-1,
    )

def get_image_caption(image):
    if image is None:
        return "No image provided."
    
    image = image.convert("RGB")

    # max_new_tokens: limit tokens to trade detail for speed
    # num beams: usually 4, for diversity; 1 for greedy decoding
    result = pipe(image,text="Detailed", max_new_tokens=32, num_beams=1, do_sample=False)
    return result[0]['generated_text']

# api w/ gradio
api = gr.Interface(
    fn=get_image_caption,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs="text"
)

api.launch(show_api=True)
