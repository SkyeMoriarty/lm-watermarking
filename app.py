# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
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

from argparse import Namespace
args = Namespace()

arg_dict = {
    'run_gradio': False,
    'demo_public': False, 
    # 'model_name_or_path': 'facebook/opt-125m',
    'model_name_or_path': 'facebook/opt-1.3b',
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'model_name_or_path': 'EleutherAI/gpt-neo-125M',
    # 'model_name_or_path': './ptuned_opt',
    # 'load_fp16' : True,
    'load_fp16': False,  # 是否加载为半精度（节省内存）
    'prompt_max_length': 64,
    'max_new_tokens': 200,  # 200
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1,  # 即不使用beam search
    'sampling_temp': 0.7,  # 控制采样多样性，越低确定性越强
    'use_gpu': True, 
    'seeding_scheme': 'hybrid',  # prf生成策略
    'gamma': 0.25, 
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_ngrams': False,  # 是否避免重复bigram
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'is_peft_model': False,
    'base_model_path': 'facebook/opt-1.3b',
}

args.__dict__.update(arg_dict)

# from p_tuning.tuning import get_ptuned_opt
#
# get_ptuned_opt(args)

from demo_watermark import main

main(args)
