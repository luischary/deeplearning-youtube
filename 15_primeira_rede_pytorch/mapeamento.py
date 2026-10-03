from pathlib import Path
import pandas as pd

images_root = Path("./data/MNIST/images/test")
dados = {"path": [], "label": []}

for pasta_classe in images_root.iterdir():
    for image_path in pasta_classe.iterdir():
        dados["path"].append("./" + image_path.as_posix())
        dados["label"].append(pasta_classe.name)

df = pd.DataFrame(dados)
print(df)
df.to_csv("./data/metadados_teste.csv", index=False)
