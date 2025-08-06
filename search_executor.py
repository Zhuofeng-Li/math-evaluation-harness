import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout
from octotools.tools.perplexity.tool import Perplexity_Tool
from concurrent.futures import TimeoutError
import json
import logging

def setup_logging():
    # logging level WARNING
    logging.basicConfig(level=logging.WARNING)
    # OpenAI logging level WARNING
    logging.getLogger('openai').setLevel(logging.WARNING)
    # requests logging level WARNING
    logging.getLogger('requests').setLevel(logging.WARNING)
    # urllib3 logging level WARNING
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    # other logging level WARNING
    logging.getLogger('pebble').setLevel(logging.WARNING)
    logging.getLogger('concurrent.futures').setLevel(logging.WARNING)

log_printed = None
setup_logging()  

def execute_code(code, model_string):
    try:
        search_tool = Perplexity_Tool(model_string=model_string)
        result = search_tool.execute(prompt=code)
        report = ''
    except Exception:
        result = ''
        report = traceback.format_exc().split('\n')[-2]
    return result, report

class PerplexitySearch:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 60,
        model_string: str = "sonar-pro"
    ) -> None:
        self.timeout_length = timeout_length
        self.model_string = model_string

    # @staticmethod
    # def execute(
    #     code,
    #     search_tool=None
        
    # ): 
    #     try:
    #         result = search_tool.execute(query=code, num_results=5)
    #         report = ''
    #     except:
    #         result = ''
    #         report = traceback.format_exc().split('\n')[-2]
    #     return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]


    def process_generation_to_code(self, gens: str):
        return [g.strip() for g in gens]
        
    def batch_apply(self, pool, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        
        # with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count(), 8)) as pool: 
        executor = partial(execute_code, model_string=self.model_string)
        future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
        iterator = future.result()

        if len(all_code_snippets) > 100:  
            progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")  
        else:  
            progress_bar = None 

        while True:
            try:
                result = next(iterator)
                all_exec_results.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print(error)
                all_exec_results.append(("", "Timeout Error"))
                timeout_cnt += 1
            except Exception as error:
                print(error)
                exit()
            if progress_bar is not None:
                progress_bar.update(1) 
        
        if progress_bar is not None:
            progress_bar.close() 

        # batch_results = []
        # for code, (res, report) in zip(all_code_snippets, all_exec_results):
        #     # post processing
        #     res, report = str(res).strip(), str(report).strip()
        #     res, report = self.truncate(res), self.truncate(report)
        #     batch_results.append((res, report))
        return all_exec_results


def _test_perplexity():
    tool = Perplexity_Tool(model_string="sonar-pro")
    # Test queries
    test_queries = [
        "Who is the father of the father of George Washington?"
        # "Compare the latest MacBook and Dell XPS specs and recommend one.",
        # "What are the top 3 tourist attractions in Tokyo?",
    ]

    # Execute the tool with test queries
    for query in test_queries:
        print(f"\nTesting query: {query}")
        try:
            execution = tool.execute(prompt=query)
            print("Generated Response:")
            print(json.dumps(execution, indent=4))
        except Exception as e:
            print(f"Execution failed: {e}")

    print("\nDone!")

def _test():
    batch_code = [
        """
        where is china?
        """
    ]

    executor = PerplexitySearch(get_answer_from_stdout=True)
    predictions = executor.batch_apply(batch_code)
    print(predictions)


if __name__ == '__main__':
    # _test_perplexity()
    _test()