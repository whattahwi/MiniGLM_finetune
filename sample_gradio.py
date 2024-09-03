### TODO: build gradio app based on sample.py
import random
import numpy
import gradio as gr
import os

def random_response(message, history):
    command = f'python sample.py --out_dir=out-915-2 --start={message}'
    os.system(command)
    with open("out-915-2/samples", "r", encoding="utf-8") as f:
        data = f.read()
    return data

gr.ChatInterface(random_response).launch()
###