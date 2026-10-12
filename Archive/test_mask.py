from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

mask_filler = pipeline("fill-mask", model="distilroberta-base", top_k=1)
res = mask_filler("This is a <mask> text with <mask> tokens.")
print("Type of res:", type(res))
print("Res:", res)
