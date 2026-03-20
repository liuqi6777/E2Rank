import json
import sys
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import mteb
import torch
from transformers import HfArgumentParser
from qwen3_embedding_model import Qwen3Embedding


logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s#%(lineno)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger('run_mteb.py')


@dataclass
class EvalArguments:
    """
    Arguments.
    """
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name for the save path"}
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(default='results', metadata={"help": "output dir of results"})
    benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name"})
    tasks: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    langs: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    only_load: bool = field(default=False, metadata={"help": ""})
    load_model: bool = field(default=False, metadata={"help": "when only_load"})

    batch_size: int = field(default=128, metadata={"help": "Will be set to `encode_kwargs`"})
    precision: str = field(default='fp16', metadata={"help": "amp_fp16,amp_bf16,fp16,bf16,fp32"})

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(',')
        if isinstance(self.langs, str):
            self.langs = self.langs.split(',')
        for name in ('model', 'encode', 'run'):
            name = name + '_kwargs'
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))


def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = mteb.get_benchmark(benchmark).tasks
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks


def get_model(model_path: str, model_name: str, precision: str = 'fp16', **kwargs):
    model = Qwen3Embedding(model_path, model_name=model_name, precision=precision, **kwargs)
    return model


def run_bright(t, model, args, **kwargs):
    # task_instructions = {}
    Instructions = {
        "aops" : "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "biology": "Given a post, retrieve relevant passages that help answer the post.",
        "earth_science": "Given a post, retrieve relevant passages that help answer the post.",
        "economics": "Given a economics post, retrieve relevant passages that help answer the post.",
        "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem.",
        "pony": "Given a question about pony program language, retrieve relevant passages that help answer the question.",
        "psychology": "Given a psychology post, retrieve relevant passages that help answer the post.",
        "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
        "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
        "robotics": "Given a robotics post, retrieve relevant passages that help answer the post.",
        "stackoverflow": "Given a stackoverflow post, retrieve relevant passages that help answer the post.",
        "sustainable_living": "Given a sustainable_living post, retrieve relevant passages that help answer the post."
    }
    encode_kwargs = args.encode_kwargs or dict()
    t.metadata.prompt = {'query': "Given a post, retrieve relevant passages that help answer the post."}
    evaluation = mteb.MTEB(tasks=[t])
    eval_splits = evaluation.tasks[0].metadata.eval_splits
    results = evaluation.run(
        model,
        output_folder=args.output_dir,
        encode_kwargs=encode_kwargs,
        eval_splits=eval_splits,
        eval_subsets=Instructions.keys(),
        **kwargs
    )


def run_eval(model, tasks: list, args: EvalArguments, **kwargs):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, 'start'):
        model.start()
        _started = True

    for t in tasks:
        if t.metadata.name == 'BrightRetrieval':
            run_bright(t, model, args, **kwargs)
            continue
        evaluation = mteb.MTEB(tasks=[t])
        
        try:
            results = evaluation.run(
                model,
                output_folder=args.output_dir,
                encode_kwargs=encode_kwargs,
                **kwargs
            )
        except Exception as e:
            print(f'meet error when running task: {t.metadata.name}. {str(e)}')
            continue

    if model is not None and _started and hasattr(model, 'stop'):
        model.stop()
    return


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.warning(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.warning(f"Args {args}")
    del parser

    tasks = get_tasks(args.tasks, args.langs, args.benchmark)
    # print('args.model_kwargs', args.model_kwargs)
    logger.warning(f"Selected {len(tasks)} tasks:\n" + '\n'.join(str(t) for t in tasks))
    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            try:
                t.load_data()
            except Exception as e:
                logger.warning(f'meet error when loading task: {t.metadata.name}. {str(e)}')
                # t.load_data(force_download=True)
            else:
                continue
            
        if not args.load_model:
            return
    model = get_model(args.model, args.model_name, precision=args.precision, **args.model_kwargs)
    if args.only_load:
        return

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(model, tasks, args, **args.run_kwargs)
    logger.warning(f"Done {len(tasks)} tasks.")
    return


if __name__ == '__main__':
    main()
