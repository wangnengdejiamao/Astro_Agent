'''
基于`youtu-graphrag/utils/call_llm_api.py`文件魔改的LLM接口，用于调用第三方DeepSeek的API
'''

import os
import time
import json
import re
import requests
from typing import Optional, Generator

from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()

def _get_env_value(*names: str, default: str = "") -> str:
    """Return the first real environment value, ignoring unresolved placeholders."""
    placeholders = {
        "",
        "your-api-key",
        "your-api-key-here",
        "your-neo4j-password",
        "${LLM_API_KEY}",
        "${OPENAI_API_KEY}",
        "${LLM_BASE_URL}",
        "${OPENAI_BASE_URL}",
        "${LLM_MODEL}",
        "${OPENAI_MODEL}",
    }
    for name in names:
        value = os.getenv(name, "")
        value = os.path.expandvars(value).strip()
        if value and value not in placeholders:
            return value
    return default

class LLMCompletionCall:
    def __init__(self):
        self.llm_model = _get_env_value("LLM_MODEL", "OPENAI_MODEL", default="deepseek-chat")
        self.llm_base_url = _get_env_value("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://api.deepseek.com")
        self.llm_api_key = _get_env_value("LLM_API_KEY", "OPENAI_API_KEY")
        if not self.llm_api_key:
            raise ValueError("LLM API key not provided")
        
        # 构建完整的API URL
        if not self.llm_base_url.endswith('/v1/chat/completions'):
            if self.llm_base_url.endswith('/'):
                self.api_url = self.llm_base_url + 'v1/chat/completions'
            else:
                self.api_url = self.llm_base_url + '/v1/chat/completions'
        else:
            self.api_url = self.llm_base_url
            
        # HTTP请求配置
        self.temperature = float(os.getenv("LLM_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.2")))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", os.getenv("OPENAI_MAX_TOKENS", "8192")))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.8"))
        self.timeout = int(os.getenv("LLM_TIMEOUT", "1200"))
        # 分离连接超时和读取超时：连接超时10秒，读取超时使用配置的timeout
        self.connect_timeout = int(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
        
        # requests session (用于连接复用)
        # 配置连接池以支持高并发
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        })
        
        # 配置连接池大小（支持高并发）
        adapter = HTTPAdapter(
            pool_connections=100,  # 连接池数量
            pool_maxsize=100,      # 每个连接池的最大连接数
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 速率限制控制配置
        self.call_delay = float(os.getenv("LLM_CALL_DELAY", "0.0"))
        self.backoff_delay = float(os.getenv("LLM_429_BACKOFF_DELAY", "1.0"))
        
        logger.info(f"LLM API initialized: {self.api_url}, model: {self.llm_model}")
        if self.call_delay > 0:
            logger.info(f"Rate limiting enabled: call_delay={self.call_delay}s, backoff_delay={self.backoff_delay}s")

    def call_api(self, content: str) -> str:
        """
        同步版本的API调用 - 线程安全
        
        Args:
            content: Prompt content
            
        Returns:
            Generated text response
        """
        api_call_start = time.time()
        logger.debug(f"[LLMTiming] call_api开始, prompt长度: {len(content)}")
        
        try:
            # 步骤1: 构建请求数据
            prepare_start = time.time()
            data = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False
            }
            prepare_time = time.time() - prepare_start
            logger.info(f"[LLM API Request] model={self.llm_model}, temperature={self.temperature}, max_tokens={self.max_tokens}")
            logger.debug(f"[LLMTiming] call_api-构建请求: {prepare_time:.4f}s")
            
            # 步骤2: 发送HTTP请求
            # 使用元组指定连接超时和读取超时：(连接超时, 读取超时)
            http_start = time.time()
            try:
                response = self.session.post(
                    self.api_url, 
                    json=data, 
                    timeout=(self.connect_timeout, self.timeout)  # (连接超时, 读取超时)
                )
                http_time = time.time() - http_start
                logger.debug(f"[LLMTiming] call_api-HTTP请求: {http_time:.4f}s (状态码: {response.status_code})")
            except requests.exceptions.ConnectTimeout as e:
                http_time = time.time() - http_start
                logger.error(f"[LLMTiming] call_api-连接超时 (耗时:{http_time:.4f}s, 连接超时设置:{self.connect_timeout}s): {e}")
                raise
            except requests.exceptions.ReadTimeout as e:
                http_time = time.time() - http_start
                logger.error(f"[LLMTiming] call_api-读取超时 (耗时:{http_time:.4f}s, 读取超时设置:{self.timeout}s): {e}")
                raise
            except requests.exceptions.Timeout as e:
                http_time = time.time() - http_start
                logger.error(f"[LLMTiming] call_api-请求超时 (耗时:{http_time:.4f}s): {e}")
                raise
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            # 步骤3: 解析响应
            parse_start = time.time()
            result = response.json()
            
            # 提取响应内容
            if 'choices' not in result or not result['choices']:
                raise Exception("API响应格式错误：缺少choices字段")
            
            # Use .get() for safe access to the 'content' key
            raw_content = result['choices'][0]['message'].get('content')
            if raw_content is None:
                raw_content = result['choices'][0]['message'].get('reasoning_content')
            if raw_content is None:
                raise Exception("API响应格式错误：缺少content字段")
            # 清理内容
            clean_completion = self._clean_llm_content(raw_content)
            parse_time = time.time() - parse_start
            logger.debug(f"[LLMTiming] call_api-解析响应: {parse_time:.4f}s, 响应长度: {len(clean_completion)}")
            
            total_time = time.time() - api_call_start
            logger.debug(f"[LLMTiming] call_api总时间: {total_time:.4f}s (构建:{prepare_time:.4f}s + HTTP请求:{http_time:.4f}s + 解析:{parse_time:.4f}s)")
            
            # 速率限制：每次成功调用后延迟，避免触发 token 上限
            if self.call_delay > 0:
                time.sleep(self.call_delay)
            
            return clean_completion
            
        except requests.exceptions.RequestException as e:
            total_time = time.time() - api_call_start
            logger.error(f"[LLMTiming] call_api失败(总时间:{total_time:.4f}s): LLM API网络请求失败: {e}")
            # 遇到 429 时额外退避
            err_str = str(e)
            if "429" in err_str or "Rate limit" in err_str:
                logger.warning(f"遇到 429 限流，退避 {self.backoff_delay}s...")
                time.sleep(self.backoff_delay)
            raise e
        except Exception as e:
            total_time = time.time() - api_call_start
            logger.error(f"[LLMTiming] call_api失败(总时间:{total_time:.4f}s): LLM API调用失败: {e}")
            raise e

    def call_api_stream(self, content: str) -> Generator[str, None, None]:
        """
        流式API调用 - 同步版本
        
        Args:
            content: Prompt content
            
        Yields:
            Generated text chunks
        """
        try:
            # 构建请求数据（启用流式）
            data = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": True
            }
            
            # 发送流式请求
            response = self.session.post(
                self.api_url,
                json=data,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            complete_content = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data: '):
                    continue
                    
                data_content = line[6:]  # 移除 "data: " 前缀
                if data_content == '[DONE]':
                    break
                    
                try:
                    chunk_data = json.loads(data_content)
                    if 'choices' in chunk_data and chunk_data['choices']:
                        delta = chunk_data['choices'][0].get('delta', {})
                        content_chunk = delta.get('content', '')
                        
                        if content_chunk:
                            complete_content += content_chunk
                            yield content_chunk
                            
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            
            logger.debug(f"流式LLM API调用完成，总长度: {len(complete_content)}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"流式LLM API网络请求失败: {e}")
            raise e
        except Exception as e:
            logger.error(f"流式LLM API调用失败: {e}")
            raise e

    def _clean_llm_content(self, text: str) -> str:
        """清理LLM响应内容"""
        if not isinstance(text, str):
            return ""
        
        # 规范化换行符
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        
        # 移除零宽字符
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        
        # 处理代码块
        fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        m = fence_re.match(t)
        if m:
            t = m.group("body").strip()
        else:
            # 处理简单的代码块标记
            if t.startswith("```") and t.endswith("```") and len(t) >= 6:
                t = t[3:-3].strip()

        # 处理JSON前缀
        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()

        return t

    def close(self):
        """关闭HTTP session"""
        if self.session:
            self.session.close()
            logger.debug("HTTP session closed")

    def __del__(self):
        """析构函数，确保session被关闭"""
        try:
            self.close()
        except:
            pass

# 使用示例
def example_usage():
    """使用示例"""
    llm_client = LLMCompletionCall()
    
    try:
        # 同步调用
        response = llm_client.call_api("你好，请介绍一下Python编程语言")
        print("同步响应:", response)
        
        # 流式调用
        print("\n流式响应:")
        for chunk in llm_client.call_api_stream("请写一首关于春天的诗"):
            print(chunk, end='', flush=True)
        print()
        
    finally:
        llm_client.close()

if __name__ == "__main__":
    example_usage()
