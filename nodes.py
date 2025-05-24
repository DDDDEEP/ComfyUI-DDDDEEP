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

LISTS_PATH = join(dirname(realpath(__file__)), "CustomConfig", "TermLists")


def _preprocess_keywords(keywords_str):
    """预处理关键词字符串
    
    Args:
        keywords_str: 输入的关键词字符串，以+或&分隔
        
    Returns:
        dict: 处理后的关键词字典，包含关键词和操作符
    """
    # 检查是否包含+或&
    if '+' in keywords_str:
        parts = keywords_str.split('+')
        operator = '+'
    elif '&' in keywords_str:
        parts = keywords_str.split('&')
        operator = '&'
    else:
        # 如果没有特殊分隔符，则视为单个关键词
        parts = [keywords_str]
        operator = None
    
    # 处理每个部分
    result = []
    for part in parts:
        part = part.strip()
        if part:  # 忽略空字符串
            result.append(part)
    
    return {'keywords': result, 'operator': operator}


# 首先添加 PromptItem 类定义
class PromptItem:
    def __init__(self, pos="", neg="", is_portrait=False):
        self.pos = pos
        self.neg = neg
        self.is_portrait = is_portrait

    @classmethod 
    def from_dict(cls, data):
        if isinstance(data, dict):
            return cls(data.get("pos", ""), data.get("neg", ""),  data.get("is_portrait", False))
        return cls(str(data), "", False)  # 处理字符串情况

    def to_dict(self):
        return {"pos": self.pos, "neg": self.neg, "is_portrait": self.is_portrait}

    def __str__(self):
        return f"PromptItem(pos='{self.pos}', neg='{self.neg}')"

class PromptTermList:
    filename = ""
    data = {"None": PromptItem()}
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
                raw_data = json.loads(f.read())
            # 统一转换为 PromptItem 对象
            cls.data = {}
            for key, value in raw_data.items():
                cls.data[key] = PromptItem.from_dict(value)
            cls.data_labels = list(cls.data.items())
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
                "allow_keywords": ("STRING", {"multiline": False, "dynamicPrompts": False, "tooltip": "输入白名单"}),
                "block_keywords": ("STRING", {"multiline": False, "dynamicPrompts": False, "tooltip": "输入黑名单"}),
            },
        }

    RETURN_TYPES = ("PROMPT_ITEM",)
    RETURN_NAMES = ("prompt_item",)
    FUNCTION = "run"

    def _get_text_by_term(self, term):
        for key, value in self.data_labels:
            if key == term:
                return (value.pos, value.neg)  # 如果是字符串，则负向为空
        return ("", "")
    
    def run(self, term, populated_text="", populated_neg_text="", enable_random=False, seed=None, allow_keywords="", block_keywords=""):
        
        term = term[:len(term)]
        
        # 获取代码执行的值
        if not enable_random:
            pos_text, neg_text = self._get_text_by_term(term)
        else:
            # 1. 先获取基础列表
            target_list = self.data_labels
            
            # 2. 应用白名单过滤
            if allow_keywords:
                keywords_data = _preprocess_keywords(allow_keywords)
                keywords_list = keywords_data['keywords']
                operator = keywords_data['operator']
                
                if operator == '&' or operator is None:  # 使用交集模式
                    # 取所有白名单结果的交集
                    filtered_lists = []
                    for keyword in keywords_list:
                        filtered = self._get_filtered_list(target_list, keyword, is_whitelist=True)
                        if filtered:
                            filtered_lists.append(filtered)
                        else:
                            return ("", "", term)
                    
                    # 取交集 - 只保留在所有过滤结果中都存在的项
                    if filtered_lists:
                        result_set = set((key, value) for key, value in filtered_lists[0])
                        for filtered in filtered_lists[1:]:
                            result_set.intersection_update((key, value) for key, value in filtered)
                        target_list = list(result_set)
                else:  # 使用合并模式 (operator == '+')
                    # 合并所有白名单结果
                    filtered_lists = []
                    for keyword in keywords_list:
                        filtered = self._get_filtered_list(target_list, keyword, is_whitelist=True)
                        if filtered:
                            filtered_lists.extend(filtered)
                    # 去重合并 - 使用键作为唯一标识
                    unique_dict = {}
                    for key, value in filtered_lists:
                        unique_dict[key] = value
                    target_list = list(unique_dict.items())
                
                if not target_list:
                    return ("", "", term)
                    
            # 3. 应用黑名单过滤
            if block_keywords:
                keywords_data = _preprocess_keywords(block_keywords)
                keywords_list = keywords_data['keywords']
                
                # 黑名单始终是依次过滤，不考虑操作符
                for keyword in keywords_list:
                    target_list = self._get_filtered_list(target_list, keyword, is_whitelist=False)
                    if not target_list:
                        return ("", "", term)
                
            # 4. 随机选择结果
            index = seed % len(target_list)
            res = target_list[index]
            pos_text = res[1].pos
            neg_text = res[1].neg
            
        # 使用强制值覆盖对应项
        final_pos = populated_text if populated_text else pos_text
        final_neg = populated_neg_text if populated_neg_text else neg_text
        
        # 创建新的 PromptItem 对象并返回
        prompt_item = self.data[term]  # 获取原始对象以保留 is_portrait
        prompt_item.pos = final_pos    # 更新 pos
        prompt_item.neg = final_neg    # 更新 neg
        return (prompt_item,)         # 直接返回对象
   
        
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
            value_lower = value.pos.lower()
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
            self.data[label] = PromptItem.from_dict(value)
        with io.open(join(LISTS_PATH, "{}.json".format(self.filename)), mode="w",
                     encoding="utf-8") as f:
            # 转换为可序列化的字典
            serializable_data = {k: v.to_dict() for k, v in self.data.items()}
            json.dump(serializable_data, f, indent=4)


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



class PromptItemCollection:

    @classmethod
    def INPUT_TYPES(cls):

        
        return {
            "required": {
                "force_scale": (["None", "Portrait", "Landscape"],),
                "long": ("INT", {"default": 1216, "min": 1, "max": 100000}),
                "short": ("INT", {"default": 832, "min": 1, "max": 100000}),
                "prompt_item_1": ("PROMPT_ITEM",),
            },
            "optional": {
                "prompt_item_2": ("PROMPT_ITEM",),
                "prompt_item_3": ("PROMPT_ITEM",),
                "prompt_item_4": ("PROMPT_ITEM",),
                "prompt_item_5": ("PROMPT_ITEM",),
                "prompt_item_6": ("PROMPT_ITEM",),
                "prompt_item_7": ("PROMPT_ITEM",),
                "prompt_item_8": ("PROMPT_ITEM",),
                "prompt_item_9": ("PROMPT_ITEM",),
                "prompt_item_10": ("PROMPT_ITEM",),
                "prompt_item_11": ("PROMPT_ITEM",),
                "prompt_item_12": ("PROMPT_ITEM",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "pos", "neg")
    CATEGORY = "DDDDEEP/PromptCollection"
    FUNCTION = "run"

    def process_alias(self, text):
        """处理 alias 映射
        Args:
            text: 原始文本
        Returns:
            str: 处理后的文本
        """
        # 分割并清理输入文本
        words = [w.strip() for w in text.split(',') if w.strip()]
        processed_words = []
        
        # 预先读取 alias 映射文件
        alias_map = {}
        alias_map_path = join(dirname(realpath(__file__)), "CustomConfig", "AliasMap.json")
        try:
            with open(alias_map_path, 'r', encoding='utf-8') as f:
                alias_map = json.load(f)
        except Exception as e:
            print(f"Error loading alias map: {str(e)}")
        
        for word in words:
            if word.startswith('{alias:') and word.endswith('}'): 
                # 提取 alias 键名
                alias_key = word[7:-1]
                if alias_key in alias_map:
                    # 将值分割并添加到处理后的单词列表
                    alias_values = [v.strip() for v in alias_map[alias_key].split(',')]
                    processed_words.extend(alias_values)
                else:
                    processed_words.append(word)  # 如果找不到映射，保留原始文本
            else:
                processed_words.append(word)
                
        return ', '.join(processed_words)

    def process_neg(self, pos_prompt, neg_text):
        """处理 neg 映射
        Args:
            pos_prompt: 正向提示词
            neg_text: 负向提示词
        Returns:
            tuple: (处理后的正向提示词, 处理后的负向提示词)
        """
        if not neg_text:
            return pos_prompt, neg_text

        # 预先读取 negative 映射文件
        neg_map = {}
        neg_map_path = join(dirname(realpath(__file__)), "CustomConfig", "NegativeMap.json")
        try:
            with open(neg_map_path, 'r', encoding='utf-8') as f:
                neg_map = json.load(f)
        except Exception as e:
            print(f"Error loading negative map: {str(e)}")
            return pos_prompt, neg_text

        # 处理负向提示词，收集需要删除的单词
        words_to_remove = set()
        neg_prompts = []
        
        # 按原始格式分割负向提示词
        for neg in neg_text.split(','):
            neg = neg.strip()
            if neg.startswith('{neg:') and neg.endswith('}'): 
                neg_key = neg[5:-1]
                if neg_key in neg_map:
                    # 收集需要删除的单词
                    remove_words = [w.strip() for w in neg_map[neg_key].split(',')]
                    words_to_remove.update(remove_words)
                else:
                    neg_prompts.append(neg)
            else:
                neg_prompts.append(neg)

        # 处理正向提示词，保持原有格式
        if words_to_remove:
            # 按行分割，保留换行符
            lines = pos_prompt.split('\n')
            processed_lines = []
            
            for line in lines:
                # 处理每一行的词语
                words = [w.strip() for w in line.split(',')]
                filtered_words = []
                
                for word in words:
                    should_keep = True
                    word_parts = word.split()
                    for remove_word in words_to_remove:
                        if remove_word in word_parts:
                            should_keep = False
                            break
                    if should_keep:
                        filtered_words.append(word)
                
                # 重建这一行，保持逗号分隔
                if filtered_words:
                    processed_lines.append(','.join(filtered_words))
                else:
                    processed_lines.append('')  # 保持空行
            
            # 重建完整文本，保持换行符
            final_pos = '\n'.join(processed_lines)
        else:
            final_pos = pos_prompt

        # 重建负向提示词，保持原有格式
        final_neg = ','.join(neg_prompts)

        return final_pos, final_neg

    def run(self, force_scale, long, short, prompt_item_1, **kwargs):
        # 收集所有有效的 prompt items
        
        
        
        
        items = [prompt_item_1]
        for i in range(2, 13):
            item = kwargs.get(f"prompt_item_{i}")
            if item is not None:
                items.append(item)
        
        # 合并所有的 pos 和 neg
        pos_prompts = []
        neg_prompts = []
        has_portrait = False
        
        for item in items:
            if item.pos:  # 处理 pos
                processed_pos = self.process_alias(item.pos)
                pos_prompts.append(processed_pos)
            if item.neg:  # 处理 neg
                neg_prompts.append(item.neg)
            if item.is_portrait:
                has_portrait = True
        
        # 合并所有 prompt
        final_pos = ',\n'.join(filter(None, pos_prompts))
        final_neg = ',\n'.join(filter(None, neg_prompts))
        
        # 处理 neg 映射
        if final_neg:
            final_pos, final_neg = self.process_neg(final_pos, final_neg)
        
        # 决定输出尺寸
        if force_scale == "Portrait":
            return (short, long, final_pos, final_neg)
        elif force_scale == "Landscape":
            return (long, short, final_pos, final_neg)
        else:
            if has_portrait:
                return (short, long, final_pos, final_neg)
            return (long, short, final_pos, final_neg)

NODE_CLASS_MAPPINGS["PromptItemCollection"] = PromptItemCollection
NODE_DISPLAY_NAME_MAPPINGS["PromptItemCollection"] = f"PromptItemCollection/{__author__}"
