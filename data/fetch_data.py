import os
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import datasets
import argparse


class DatasetProcessor(ABC):
    """Base class for dataset processors"""

    def __init__(self, dataset_name: str, output_dir: str, split: str = "test", sample_size: int = None):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.split = split
        self.sample_size = sample_size
        self.output_file = os.path.join(output_dir, f"{split}.jsonl")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single example"""
        pass

    def load_dataset(self) -> datasets.Dataset:
        """Load dataset from Hugging Face"""
        try:
            print(f"Loading dataset: {self.dataset_name} ({self.split})")
            dataset = datasets.load_dataset(self.dataset_name, split=self.split)
            print(f"  - Successfully loaded '{self.split}', num samples: {len(dataset)}")

            # Sample if needed
            if self.sample_size and len(dataset) > self.sample_size:
                dataset = dataset.shuffle(seed=42).select(range(self.sample_size))
                print(f"  - Sampled {self.sample_size} examples")

            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please check your Hugging Face login and access permissions.")
            raise

    def process(self) -> None:
        """Process and save the dataset"""
        try:
            dataset = self.load_dataset()
            formatted_dataset = dataset.map(self.format_example, remove_columns=dataset.column_names)
            formatted_dataset.to_json(self.output_file, lines=True)
            print(f"\nDataset saved to {self.output_file}")
            print(f"All samples: {len(formatted_dataset)}")
        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise


class AIME25Processor(DatasetProcessor):
    """Processor for AIME25 dataset"""

    def __init__(self, output_dir: str = "./data/aime25", split: str = "test", sample_size: int = None):
        super().__init__("math-ai/aime25", output_dir, split, sample_size)

    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": example['problem'],
            "answer": example['answer']
        }


class MMLUProProcessor(DatasetProcessor):
    """Processor for MMLU-Pro dataset"""

    def __init__(self, output_dir: str = "./data/mmlu_pro", split: str = "test", sample_size: int = 300):
        super().__init__("TIGER-Lab/MMLU-Pro", output_dir, split, sample_size)

    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example['question']
        options = example['options']
        cot = example.get("cot_content", "")

        formatted_options = []
        for i, option in enumerate(options):
            if option == "N/A":
                break
            formatted_options.append(f"{chr(65 + i)}. {option}")

        formatted_question = f"{question}\n Options are:\n" + "\n".join(formatted_options)

        return {
            "question": formatted_question,
            "cot": cot,
            "answer": example['answer']
        }


class SuperGPQAProcessor(DatasetProcessor):
    """Processor for SuperGPQA dataset"""

    def __init__(self, output_dir: str = "./data/supergpqa", split: str = "train", sample_size: int = 300):
        super().__init__("m-a-p/SuperGPQA", output_dir, split, sample_size)

    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example['question']
        options = example['options']

        formatted_options = []
        for i, option in enumerate(options):
            if option == "N/A":
                break
            formatted_options.append(f"{chr(65 + i)}. {option}")

        formatted_question = f"{question}\n Options are:\n" + "\n".join(formatted_options)
        correct_answer = example['answer_letter'] + ". " + example['answer']

        return {
            "question": formatted_question,
            "answer": correct_answer
        }


class GPQAProcessor(DatasetProcessor):
    """Processor for GPQA dataset"""

    def __init__(self, output_dir: str = "./data/gpqa_diamond", split: str = "train", sample_size: int = None):
        super().__init__("Idavidrein/gpqa", output_dir, split, sample_size, config_name="gpqa_diamond")
        # Note: We need to override load_dataset to handle config_name

    def load_dataset(self) -> datasets.Dataset:
        try:
            print(f"Loading dataset: {self.dataset_name} (gpqa_diamond, {self.split})")
            dataset = datasets.load_dataset(self.dataset_name, "gpqa_diamond", split=self.split)
            print(f"  - Successfully loaded '{self.split}', num samples: {len(dataset)}")

            # Select and rename columns
            dataset = dataset.select_columns(["Question", "Correct Answer"])
            dataset = dataset.rename_columns({
                "Question": "question",
                "Correct Answer": "answer"
            })

            # Process questions
            def process_question(example):
                question = example['question']
                if '\n\n' in question:
                    example['question'] = question.split('\n\n', 1)[1]
                return example

            dataset = dataset.map(process_question)

            # Sample if needed
            if self.sample_size and len(dataset) > self.sample_size:
                dataset = dataset.shuffle(seed=42).select(range(self.sample_size))
                print(f"  - Sampled {self.sample_size} examples")

            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please check your Hugging Face login and access permissions.")
            raise

    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # GPQA is already formatted in load_dataset
        return example


class OctoSearchProcessor(DatasetProcessor):
    """Processor for OctoSearch datasets"""

    def __init__(self, output_dir: str = "./data", template_type: str = "search_r1", random_sample: bool = False, sample_size: int = 500):
        # This is a special processor that handles multiple datasets
        self.template_type = template_type
        self.random_sample = random_sample
        self.sample_size = sample_size
        super().__init__("RUC-NLPIR/FlashRAG_datasets", output_dir, split="train")

    def make_prefix(self, dp):
        question = dp['question']

        if self.template_type == 'search_r1':
            prefix = (
                "Answer the given question. You must conduct reasoning inside <audio> and <audio> "
                "first every time you get new information. After reasoning, if you find you lack "
                "some knowledge, you can call a search engine by <search> query </search> "
                "and it will return the top searched results between <information> and "
                "</information>. You can search as many times as your want. If you find no "
                "further external knowledge needed, you can directly provide the answer inside "
                "<audio> and <audio>, without detailed illustrations. For example, "
                "鸣巍 Beijing 鸣巍. Question: {question}"
            )
            return prefix.format(question=question)
        elif self.template_type == 'octo_search_r1':
            prefix = (
                "Answer the given question by calling the Search tool. "
                "You must perform your reasoning within  francès and  francès before each tool call. "
                "After reasoning, call the Search tool (described as: a tool that performs web search based on a given text query) "
                "by passing the query inside <tool_query>...</tool_query>. The tool will return its result between <tool_result> and </tool_result>. "
                "You may call the tool as many times as needed. If no further tool calls are required, "
                "provide the final answer directly within  francès... francès, without additional explanation. "
                "For example:  francès Beijing  francès. Question: {question}"
            )
            return prefix.format(question=question)
        else:
            raise NotImplementedError(f"Template type {self.template_type} not supported")

    def random_sample_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        if self.random_sample and len(dataset) > self.sample_size:
            indices = random.sample(range(len(dataset)), self.sample_size)
            return dataset.select(indices)
        return dataset

    def process(self) -> None:
        try:
            # Process NQ dataset
            print("Processing NQ dataset...")
            nq_dataset = datasets.load_dataset(self.dataset_name, 'nq')
            nq_train = nq_dataset['train']
            nq_test = nq_dataset['test']

            # Process function
            def make_map_fn(split: str, data_source: str):
                def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
                    example['question'] = example['question'].strip()
                    if example['question'][-1] != '?':
                        example['question'] += '?'
                    question = self.make_prefix(example)

                    return {
                        "problem": example['question'],
                        "answer": example['golden_answers'][0],
                        "idx": idx
                    }

                return process_fn

            # Process and save NQ dataset
            nq_train_processed = nq_train.map(make_map_fn('train', 'nq'), with_indices=True, remove_columns=nq_train.column_names)
            nq_test_processed = nq_test.map(make_map_fn('test', 'nq'), with_indices=True, remove_columns=nq_test.column_names)

            nq_dir = os.path.join(self.output_dir, 'nq')
            os.makedirs(nq_dir, exist_ok=True)
            nq_train_processed.to_json(os.path.join(nq_dir, 'train.jsonl'), lines=True)
            nq_test_processed.to_json(os.path.join(nq_dir, 'test.jsonl'), lines=True)

            # Process other datasets
            datasets_info = [
                ('hotpotqa', 'dev'),
                ('2wikimultihopqa', 'dev'),
                ('musique', 'dev'),
                ('bamboogle', 'test')
            ]

            for ds_name, ds_split in datasets_info:
                print(f"Processing {ds_name} dataset...")
                ds = datasets.load_dataset(self.dataset_name, ds_name)
                ds_test = ds[ds_split]
                ds_test_processed = ds_test.map(make_map_fn('test', ds_name), with_indices=True, remove_columns=ds_test.column_names)

                if self.random_sample:
                    ds_test_processed = self.random_sample_dataset(ds_test_processed)

                ds_dir = os.path.join(self.output_dir, ds_name)
                os.makedirs(ds_dir, exist_ok=True)
                ds_test_processed.to_json(os.path.join(ds_dir, 'test.jsonl'), lines=True)

            print("All OctoSearch datasets processed successfully!")

        except Exception as e:
            print(f"Error processing OctoSearch datasets: {e}")
            raise


class DatasetProcessorFactory:
    """Factory for creating dataset processors"""

    _processors = {}

    @classmethod
    def register_processor(cls, name: str, processor_class):
        """Register a new dataset processor"""
        cls._processors[name] = processor_class

    @classmethod
    def create_processor(cls, name: str, **kwargs):
        """Create a dataset processor by name"""
        if name not in cls._processors:
            raise ValueError(f"Processor '{name}' not found. Available processors: {list(cls._processors.keys())}")
        return cls._processors[name](**kwargs)

    @classmethod
    def list_processors(cls):
        """List all available processors"""
        return list(cls._processors.keys())


# Register processors
DatasetProcessorFactory.register_processor('aime25', AIME25Processor)
DatasetProcessorFactory.register_processor('mmlu_pro', MMLUProProcessor)
DatasetProcessorFactory.register_processor('supergpqa', SuperGPQAProcessor)
DatasetProcessorFactory.register_processor('gpqa', GPQAProcessor)
DatasetProcessorFactory.register_processor('octo_search', OctoSearchProcessor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for math evaluation')
    parser.add_argument('--processor', type=str, required=True, help='Name of the dataset processor to use')
    parser.add_argument('--output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to process')
    parser.add_argument('--sample_size', type=int, help='Number of samples to randomly extract')
    parser.add_argument('--template_type', type=str, default='search_r1', help='Template type for octo_search processor')
    parser.add_argument('--random_sample', action='store_true', help='Whether to randomly sample a subset of the dataset (for octo_search)')

    args = parser.parse_args()

    # Create processor arguments
    processor_kwargs = {}
    if args.output_dir:
        processor_kwargs['output_dir'] = args.output_dir
    if args.split:
        processor_kwargs['split'] = args.split
    if args.sample_size:
        processor_kwargs['sample_size'] = args.sample_size
    if args.processor == 'octo_search':
        processor_kwargs['template_type'] = args.template_type
        processor_kwargs['random_sample'] = args.random_sample

    # Create and run processor
    try:
        processor = DatasetProcessorFactory.create_processor(args.processor, **processor_kwargs)
        processor.process()
    except Exception as e:
        print(f"Error: {e}")

"""

python data/fetch_data.py --processor aime25

python data/fetch_data.py --processor mmlu_pro --output_dir ./data/mmlu_pro_new

python data/fetch_data.py --processor supergpqa --sample_size 500

python data/fetch_data.py --processor octo_search --template_type octo_search_r1 --random_sample

"""