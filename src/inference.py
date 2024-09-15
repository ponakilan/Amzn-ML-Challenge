import ollama
import easyocr
import pickle

class Inferencer:
    def __init__(self, pkl_path):
        self.reader = easyocr.Reader(['en'])
        self.prompt = pickle.load(open(pkl_path, "rb"))

    def get_response(self, image_link, entity_name):
        result = self.reader.readtext(image_link)
        result = " ".join([detection[1] for detection in result])
        res = ollama.generate(model='llama3.1', prompt=self.prompt.format(result, entity_name))['response']
        return result, res
