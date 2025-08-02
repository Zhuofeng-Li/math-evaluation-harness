from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

ds = ds["train"]

ds = ds.select_columns(["Question", "Correct Answer"])

ds = ds.rename_columns({
    "Question": "question",
    "Correct Answer": "answer"
})

# TODO: mkdir 
ds.to_json("test.jsonl", lines=True)