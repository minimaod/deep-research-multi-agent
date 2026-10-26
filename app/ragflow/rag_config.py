"""
RAGFlow 连接配置加载模块

集中读取 RAGFlow SDK 需要的 API Key 和服务地址，供原始调用示例与
LangChain 工具共用。这样后续如果 .env 字段或读取规则调整，只需要改这一处。
"""

import os
import time
from typing import Optional, Tuple

from dotenv import find_dotenv, load_dotenv


def _load_ragflow_env() -> Tuple[Optional[str], Optional[str]]:
    """
    加载 RAGFlow 环境变量

    使用 python-dotenv 自动向上查找 .env，保持和项目其他配置加载方式一致。
    :return: (api_key, base_url)，缺失配置时对应位置返回 None
    """
    load_dotenv(find_dotenv())

    # RAGFlow SDK 初始化只需要这两个核心字段：认证 API Key 和服务基础地址
    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = os.getenv("RAGFLOW_API_URL")
    return api_key, base_url


def get_ragflow_client(timeout=20):
    """Per-call SDK client with bounded HTTP requests; no global monkey-patch."""
    import requests
    from ragflow_sdk import RAGFlow
    from app.harness.schemas import ErrorCode
    from app.harness.tool_wrapper import ToolFailure
    api_key, base_url = _load_ragflow_env()
    if not api_key or not base_url:
        raise ToolFailure(ErrorCode.UNAVAILABLE, "missing_ragflow_configuration")

    class BoundedRAGFlow(RAGFlow):
        def __init__(self):
            super().__init__(api_key, base_url)
            self.deadline = time.monotonic() + timeout
            self.http = requests.Session()

        def request(self, method, path, **kwargs):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ragflow deadline")
            response = self.http.request(method, self.api_url + path,
                headers=self.authorization_header, timeout=(min(3, remaining), remaining), **kwargs)
            response.raise_for_status()
            return response

        def post(self, path, json=None, stream=False, files=None):
            return self.request("POST", path, json=json, stream=stream, files=files)

        def get(self, path, params=None, json=None):
            return self.request("GET", path, params=params, json=json)

        def delete(self, path, json):
            return self.request("DELETE", path, json=json)

        def close(self):
            self.http.close()

    return BoundedRAGFlow()
