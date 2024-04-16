import gradio as gr
from src.pipeline.predict import generate_caption

demo = gr.Interface(fn=generate_caption,
             inputs=gr.Image(),
             outputs=[gr.Textbox(label="Generated Caption", lines=3)],
             )
demo.launch()