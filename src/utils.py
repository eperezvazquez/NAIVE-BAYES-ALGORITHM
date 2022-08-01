
import pandas as pd


class Helpers():
    @classmethod
    # función para convertir región a numérico
    def normalize_str(text_string):
        if text_string is not None:
             result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
        else:
             result=None 
        return result