from src.models.predict import predict_from_dict
import pandas as pd

p = "data/processed/heart_disease_processed.csv"
df = pd.read_csv(p)
row = df.iloc[0].to_dict()
for k in ["id", "dataset", "num", "target"]:
    row.pop(k, None)

print('Input row (sample):')
print(row)
print('\nPrediction result:')
print(predict_from_dict(row))
