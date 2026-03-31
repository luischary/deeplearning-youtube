import pandas as pd

from src.tokenizer import BPETokenizer

base = pd.read_parquet("data/wiki_en/wiki_en_sample.pq")
base = base[base["split"] == "train"].sample(100_000).reset_index(drop=True)
tokenizer = BPETokenizer(vocab_size=5_000)
tokenizer.train(base.text.tolist(), vocab_size=5000)
tokenizer.save("artifacts/tokenizer_wiki_en_5k.json")
