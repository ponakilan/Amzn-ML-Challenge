import ollama
import easyocr
from .lora import LoraModel 

class Inferencer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def get_response(self, image_link, prompt, entity_name):
        result = self.reader.readtext(image_link)
        result = " ".join([detection[1] for detection in result])
        res = ollama.generate(model='llama3.1', prompt=LoraModel.prompt.format(result, entity_name))['response']
        return res
