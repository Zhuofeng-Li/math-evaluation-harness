# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import datasets
import random
from typing import Any, Dict, List

import argparse

def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'search_r1':
        """This works for any base model"""
        prefix = (
            "Answer the given question. You must conduct reasoning inside <think> and </think> "
            "first every time you get new information. After reasoning, if you find you lack "
            "some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and "
            "</information>. You can search as many times as your want. If you find no "
            "further external knowledge needed, you can directly provide the answer inside "
            "<answer> and </answer>, without detailed illustrations. For example, "
            "<answer> Beijing </answer>. Question: {question}"
        )
        prefix = prefix.format(question=question)
    elif template_type == 'octo_search_r1':
        prefix = (
            "Answer the given question by calling the Search tool. \
            You must perform your reasoning within <think> and </think> before each tool call. \
            After reasoning, call the Search tool (described as: a tool that performs web search based on a given text query) \
            by passing the query inside <tool_query>...</tool_query>. The tool will return its result between <tool_result> and </tool_result>. \
            You may call the tool as many times as needed. If no further tool calls are required, \
            provide the final answer directly within <answer>...</answer>, without additional explanation. \
            For example: <answer> Beijing </answer>. Question: {question}"
        )
        prefix = prefix.format(question=question)
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--random_sample', action='store_true', help='Whether to randomly sample a subset of the dataset')
    parser.add_argument('--sample_size', type=int, default=500, help='Number of samples to randomly extract')

    args = parser.parse_args()

    # Load NQ dataset
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
    
    # Extract train and test splits
    train_dataset = dataset['train'] 
    test_dataset = dataset['test']  

    # add a row to each data item that represents a unique id
    def make_map_fn(split: str, data_source: str):
        def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)

            # data = {
            #     "data_source": data_source,
            #     "prompt": [{
            #         "role": "user",
            #         "content": question,
            #     }],
            #     "ability": "fact-reasoning",
            #     "reward_model": {
            #         "style": "rule",
            #         "ground_truth": example['golden_answers']
            #     },
            #     "extra_info": {
            #         'split': split,
            #         'index': idx,
            #     }
            # }

            data = {
                "problem": example['question'],
                "answer": example['golden_answers'][0],
                "idx": idx
            }
            
            return data

        return process_fn

    def random_sample(dataset: datasets.Dataset, sample_size: int) -> datasets.Dataset:
        if len(dataset) > sample_size:
            indices = random.sample(range(len(dataset)), sample_size)
            return dataset.select(indices)
        return dataset

    # Process NQ datasets
    train_dataset = train_dataset.map(function=make_map_fn('train', 'nq'), with_indices=True, remove_columns=train_dataset.column_names)  
    test_dataset = test_dataset.map(function=make_map_fn('test', 'nq'), with_indices=True, remove_columns=test_dataset.column_names)    # type: ignore

    # Load and process multiple datasets for testing, with 500 random samples
    
    # HotpotQA dataset
    hotpotqa_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa')
    hotpotqa_test_dataset = hotpotqa_dataset['dev'] # type: ignore
    hotpotqa_test_dataset = hotpotqa_test_dataset.map(function=make_map_fn('test', 'hotpotqa'), with_indices=True, remove_columns=hotpotqa_test_dataset.column_names)  # type: ignore

    # 2wikimultihopqa dataset
    two_wikimultihopqa_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', '2wikimultihopqa')
    two_wikimultihopqa_test_dataset = two_wikimultihopqa_dataset['dev'] # type: ignore         
    two_wikimultihopqa_test_dataset = two_wikimultihopqa_test_dataset.map(function=make_map_fn('test', '2wikimultihopqa'), with_indices=True, remove_columns=two_wikimultihopqa_test_dataset.column_names)  # type: ignore

    # musique dataset
    musique_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'musique')
    musique_test_dataset = musique_dataset['dev'] # type: ignore
    musique_test_dataset = musique_test_dataset.map(function=make_map_fn('test', 'musique'), with_indices=True, remove_columns=musique_test_dataset.column_names)  # type: ignore

    # bamboogle dataset
    bamboogle_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'bamboogle')
    bamboogle_test_dataset = bamboogle_dataset['test'] # type: ignore
    bamboogle_test_dataset = bamboogle_test_dataset.map(function=make_map_fn('test', 'bamboogle'), with_indices=True, remove_columns=bamboogle_test_dataset.column_names)  # type: ignore

    if args.random_sample:
        hotpotqa_test_dataset = random_sample(hotpotqa_test_dataset, args.sample_size)
        two_wikimultihopqa_test_dataset = random_sample(two_wikimultihopqa_test_dataset, args.sample_size)
        musique_test_dataset = random_sample(musique_test_dataset, args.sample_size)
        bamboogle_test_dataset = random_sample(bamboogle_test_dataset, args.sample_size)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    print(train_dataset[0])

    # Save processed datasets
    nq_dir = os.path.join(local_dir, 'nq')
    os.makedirs(nq_dir, exist_ok=True)
    train_dataset.to_json(os.path.join(nq_dir, 'train.jsonl'), lines=True)  # type: ignore
    test_dataset.to_json(os.path.join(nq_dir, 'test.jsonl'), lines=True)    # type: ignore

    # HotpotQA dataset
    hotpotqa_dir = os.path.join(local_dir, 'hotpotqa')
    os.makedirs(hotpotqa_dir, exist_ok=True)
    hotpotqa_test_dataset.to_json(os.path.join(hotpotqa_dir, 'test.jsonl'), lines=True)  # type: ignore

    # 2wikimultihopqa dataset
    two_wiki_dir = os.path.join(local_dir, '2wikimultihopqa')
    os.makedirs(two_wiki_dir, exist_ok=True)
    two_wikimultihopqa_test_dataset.to_json(os.path.join(two_wiki_dir, 'test.jsonl'), lines=True)    # type: ignore

    # musique dataset
    musique_dir = os.path.join(local_dir, 'musique')
    os.makedirs(musique_dir, exist_ok=True)
    musique_test_dataset.to_json(os.path.join(musique_dir, 'test.jsonl'), lines=True)    # type: ignore

    # bamboogle dataset
    bamboogle_dir = os.path.join(local_dir, 'bamboogle')
    os.makedirs(bamboogle_dir, exist_ok=True)
    bamboogle_test_dataset.to_json(os.path.join(bamboogle_dir, 'test.jsonl'), lines=True)    # type: ignore


"""
# octo search r1 format 
python data/octo_search.py --local_dir ./data --template_type octo_search_r1
"""