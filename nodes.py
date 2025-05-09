import io
import re
import sys
import json
import os
from os.path import realpath, join, dirname, exists
from os import listdir
import random
import numpy as np
import importlib.util
# from .utility.utility import __termlists_data_suffix__

TERMLIST_SUFFIX = '.json'

MANIFEST = {"name": "DDDDEEP Nodes",
            "version": (1, 0, 0),
            "author": "DDDDEEP",
            "project": "https://github.com/DDDDEEP/ComfyUI-DDDDEEP",
            "description": "Nodes for ComfyUI",
            "license": "MIT",
            }
__author__ = "DDDDEEP"
__version__ = "1.0.0"

LISTS_PATH = join(dirname(realpath(__file__)), "TermLists")


def _preprocess_keywords(keywords_str):
    """预处理关键词字符串
    
    Args:
        keywords_str: 输入的关键词字符串，以逗号分隔
        
    Returns:
        list: 处理后的关键词列表
    """
    # 1. 分割字符串
    parts = keywords_str.split(',')
    # 2. 处理每个部分
    result = []
    for part in parts:
        part = part.strip()
        if part:  # 忽略空字符串
            result.append(part)
    return result


class PromptTermList:
    filename = ""
    data = {"None": ""}
    data_labels = []
    allow_keyword_data = {}
    _file_mtimes = {}  # 存储文件修改时间
    has_error = False
    input_error = ("Trying to store invalid input!\nUse the format:\n"
                   "label=... ...\nvalue=.... .... ...")

    def __init_subclass__(cls, **kwargs):
        cls_name = cls.__name__
        if cls_name.endswith("List"):
            cls.filename = cls_name[:-4]
        else:
            cls.filename = cls_name

    def __init__(self):
        super(PromptTermList, self).__init__()
        self.name = type(self).__name__

    @classmethod
    def load_data_from_file(cls, file_path):
        try:
            with io.open(file_path, mode="r", encoding="utf-8") as f:
                cls.data = json.loads(f.read())
            cls.data_labels = list(cls.data.items())
            # 更新文件修改时间
            cls._file_mtimes[file_path] = os.path.getmtime(file_path)
        except (FileNotFoundError, AttributeError, json.JSONDecodeError) as e:
            print(f"Error loading data for {cls.filename}: {str(e)}")
            pass

    @classmethod
    def should_reload(cls, file_path):
        if not exists(file_path):
            return True
        last_mtime = cls._file_mtimes.get(file_path, 0)
        current_mtime = os.path.getmtime(file_path)
        return current_mtime != last_mtime

    # noinspection PyMethodParameters
    @classmethod
    def INPUT_TYPES(cls):
        list_path = join(LISTS_PATH, f"{cls.filename}{TERMLIST_SUFFIX}")
        if cls.should_reload(list_path):
            cls.load_data_from_file(list_path)
            cls.allow_keyword_data={}
        term_list = [i[0] for i in cls.data_labels]
        return {
            "required": {
                "term": (term_list,),
                "populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "可输入强制Prompt"}),
                "populated_neg_text": ("STRING", {"multiline": False, "dynamicPrompts": False, "tooltip": "可输入强制负向Prompt"}),
                "enable_random": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False", "tooltip": "Enable random seed for terms."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Determines the random seed to be used for wildcard processing."}),
                "allow_keywords": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "输入白名单"}),
                "block_keywords": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "输入黑名单"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING","STRING")
    RETURN_NAMES = ("pos_prompt", "neg_prompt", "term")
    # OUTPUT_NODE = True
    CATEGORY = "DDDDEEP/Term Nodes"
    FUNCTION = "run"

    def _get_text_by_term(self, term):
        for key, value in self.data_labels:
            if key == term:
                # 处理新的 JSON 结构
                if isinstance(value, dict):
                    return value.get("pos", ""), value.get("neg", "")
                return value, ""  # 如果是字符串，则负向为空
        return "", ""

    def run(self, term, populated_text="", populated_neg_text="", enable_random=False, seed=None, allow_keywords="", block_keywords=""):
        term = term[:len(term)]
        
        # 获取代码执行的值
        if not enable_random:
            pos_text, neg_text = self._get_text_by_term(term)
        else:
            # 1. 先获取基础列表
            target_list = self.data_labels
            
            # 2. 应用白名单过滤 - 合并所有白名单结果
            if allow_keywords:
                keywords_list = _preprocess_keywords(allow_keywords)
                filtered_lists = []
                for keyword in keywords_list:
                    filtered = self._get_filtered_list(target_list, keyword, is_whitelist=True)
                    if filtered:
                        filtered_lists.extend(filtered)
                # 去重合并
                target_list = list(dict.fromkeys(filtered_lists))
                if not target_list:
                    return ("", "", term)
                    
            # 3. 应用黑名单过滤 - 依次过滤
            if block_keywords:
                keywords_list = _preprocess_keywords(block_keywords)
                for keyword in keywords_list:
                    target_list = self._get_filtered_list(target_list, keyword, is_whitelist=False)
                    if not target_list:
                        return ("", "", term)
                
            # 4. 随机选择结果
            index = seed % len(target_list)
            res = target_list[index]
            if isinstance(res[1], dict):
                pos_text = res[1].get("pos", "")
                neg_text = res[1].get("neg", "")
            else:
                pos_text = res[1]
                neg_text = ""
        
        # 使用强制值覆盖对应项
        final_pos = populated_text if populated_text else pos_text
        final_neg = populated_neg_text if populated_neg_text else neg_text
        
        return (final_pos, final_neg, term)
   
        
    def _get_filtered_list(self, source_list, keywords, is_whitelist=True):
        """统一的过滤方法
        
        Args:
            source_list: 源数据列表
            keywords: 关键词
            is_whitelist: True表示白名单模式，False表示黑名单模式
        """
        if not keywords:
            return source_list
            
        # 缓存键需要包含过滤模式
        cache_key = f"{'allow' if is_whitelist else 'block'}:{keywords}"
        if cache_key in self.allow_keyword_data:
            return self.allow_keyword_data[cache_key]
            
        # 过滤逻辑
        filtered_list = []
        for key, value in source_list:
            key_lower = key.lower()
            value_lower = value.lower()
            keywords_lower = keywords.lower()
            
            # 检查是否匹配关键词
            is_match = (keywords_lower in key_lower or keywords_lower in value_lower)
            
            # 根据白名单/黑名单模式决定是否保留
            if (is_whitelist and is_match) or (not is_whitelist and not is_match):
                filtered_list.append((key, value))
                
        # 保存到缓存
        self.allow_keyword_data[cache_key] = filtered_list
        return filtered_list
        
    def save_data_from_input(self, text):
        """ Extracts the json values from the input text and stores them in the json file

        :type text: str
        :param text: The text input
        """
        lines = text.splitlines()
        if not len(lines) > 1:
            self.has_error = True
            print(f"{self.name}:", self.input_error)
            return
        if not all((lines[0].startswith("label="), lines[1].startswith("value="))):
            self.has_error = True
            print(f"{self.name}:", self.input_error)
            return
        label = lines[0][6:]
        value = lines[1][6:]
        if label == "None":
            print(f'{self.name}: The label "{label}" cannot be changed!')
            return
        if not value:
            if label in self.data:
                del self.data[label]
                print(f'{self.name}: The label "{label}" was deleted!')
            else:
                print(f'{self.name}: The label "{label}" does not exist!')
            return
        else:
            if label in self.data:
                print(f'{self.name}: The label "{label}" is updated!')
            else:
                print(f'{self.name}: The label "{label}" is saved!')
            self.data[label] = value
        with io.open(join(LISTS_PATH, "{}.json".format(self.filename)), mode="w",
                     encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)


class_names = [f[:-len(TERMLIST_SUFFIX)] for f in listdir(LISTS_PATH) if f.endswith(TERMLIST_SUFFIX)]
cls_list = [type(name, (PromptTermList,), {}) for name in class_names]

NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in cls_list
}

NODE_DISPLAY_NAME_MAPPINGS = {
    cls.__name__: f"{cls.__name__}/{__author__}"
    for cls in cls_list
}


# class INTConstant:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#         },
#         }
#     RETURN_TYPES = ("INT",)
#     RETURN_NAMES = ("value",)
#     FUNCTION = "get_value"
#     CATEGORY = "KJNodes/constants"

#     def get_value(self, value):
#         return (value,)


class AutoWidthHeight:

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "force_scale": (["None", "Portrait", "Landscape"],),
                "long": ("INT", {"default": 1216, "min": 1, "max": 100000}),
                "short": ("INT", {"default": 832, "min": 1, "max": 100000}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "portrait_keywords": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "输入竖图关键词"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    # OUTPUT_NODE = True
    CATEGORY = "DDDDEEP/AutoWidthHeight"
    FUNCTION = "run"

    def run(self, force_scale, long, short, prompt="", portrait_keywords=""):
        if force_scale == "Portrait":
            return (short, long)
        if force_scale == "Landscape":
            return (long, short)

        # 补全逻辑：
        # 1. 先预处理关键词数组
        # 2. 遍历关键词，判断是否在prompt中出现
        # 3. 如果出现，返回(short, long)
        # 4. 如果没有出现，返回(long, short)
        keywords_list = _preprocess_keywords(portrait_keywords)
        
        # 如果没有关键词或prompt，默认横图
        if not keywords_list or not prompt:
            return (long, short)
            
        # 转小写进行匹配
        prompt_lower = prompt.lower()
        
        # 检查是否包含任意一个竖图关键词
        for keyword in keywords_list:
            if keyword.lower() in prompt_lower:
                return (short, long)  # 竖图
                
        # 默认返回横图
        return (long, short)

NODE_CLASS_MAPPINGS["AutoWidthHeight"] = AutoWidthHeight
NODE_DISPLAY_NAME_MAPPINGS["AutoWidthHeight"] = f"AutoWidthHeight/{__author__}"


class ReturnIntSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Determines the random seed to be used for wildcard processing."}),
            },
        }

    RETURN_TYPES = ("INT:seed",)
    RETURN_NAMES = ("int_seed",)
    CATEGORY = "DDDDEEP/ReturnIntSeed"
    FUNCTION = "run"

    def run(self, seed=0):
        
        return (seed,)

NODE_CLASS_MAPPINGS["ReturnIntSeed"] = ReturnIntSeed
NODE_DISPLAY_NAME_MAPPINGS["ReturnIntSeed"] = "ReturnIntSeed/DDDDEEP"
