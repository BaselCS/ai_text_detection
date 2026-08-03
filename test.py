import pandas as pd

df = pd.read_csv("test.csv")

print([i for i in df.columns if i.startswith("bert")])
print(len([i for i in df.columns if i.startswith("bert_")]))
