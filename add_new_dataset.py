from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

ds = ds["train"]

ds = ds.select_columns(["Question", "Correct Answer"])

ds = ds.rename_columns({
    "Question": "question",
    "Correct Answer": "answer"
})

# Function to keep only content after \n\n in question
def process_question(question):
    if '\n\n' in question:
        return question.split('\n\n', 1)[1]
    return question

# Apply the processing to the question column
ds = ds.map(lambda x: {"question": process_question(x["question"])})

# TODO: mkdir 
ds.to_json("test.jsonl", lines=True)