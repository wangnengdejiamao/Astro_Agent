
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
简单脚本：测试本地 embedding_service 连接情况，并打印详细报错。

用法：
    python utils/test_embedding_api.py "自定义文本A" "自定义文本B"
"""

import os
import sys
import time
import pickle
import socket
import traceback

import numpy as np


class LocalEmbeddingServiceEncoder:
    """从 tree_comm.py 抽离的轻量版本，仅保留网络调用逻辑"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
        max_batch_size: int | None = None,
        vector_dim: int | None = None,
        max_retries: int | None = None,
    ):
        self.host = host or os.getenv("EMBEDDING_SERVICE_HOST", "localhost")
        self.port = int(port or os.getenv("EMBEDDING_SERVICE_PORT", "8035"))
        self.timeout = int(timeout or os.getenv("EMBEDDING_TIMEOUT", "600"))
        self.max_batch_size = int(max_batch_size or os.getenv("EMBEDDING_MAX_BATCH_SIZE", "8"))
        self.vector_dim = int(vector_dim or os.getenv("VECTOR_DIMENSION", "2560"))
        self.max_retries = int(max_retries or os.getenv("EMBEDDING_MAX_RETRIES", "3"))

    def encode(self, texts, batch_size: int | None = None):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._request_embeddings_with_retry(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _request_embeddings_with_retry(self, texts):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._request_embeddings(texts)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[LocalEmbeddingService] 请求失败 (尝试 {attempt}/{self.max_retries}): {exc}")
                time.sleep(1)
        raise RuntimeError(f"无法从嵌入服务获取结果: {last_error}")

    def _request_embeddings(self, texts):
        payload = pickle.dumps(texts)
        header = len(payload).to_bytes(4, byteorder="big")

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as client:
            client.sendall(header)
            client.sendall(payload)

            resp_header = self._recv_exact(client, 4)
            resp_len = int.from_bytes(resp_header, byteorder="big")
            resp_payload = self._recv_exact(client, resp_len)

        embeddings = pickle.loads(resp_payload)
        if not isinstance(embeddings, list):
            raise ValueError("嵌入服务返回格式错误")
        for emb in embeddings:
            if not isinstance(emb, (list, tuple)) or len(emb) != self.vector_dim:
                raise ValueError(f"嵌入向量维度不匹配，期望 {self.vector_dim}")
        return embeddings

    def _recv_exact(self, sock, expected_len: int) -> bytes:
        data = b""
        while len(data) < expected_len:
            chunk = sock.recv(expected_len - len(data))
            if not chunk:
                raise ConnectionError("连接中断，未收到完整数据")
            data += chunk
        return data


def run_test(texts):
    encoder = LocalEmbeddingServiceEncoder()
    print("==== Embedding API 连接测试 ====")
    print(f"目标地址: {encoder.host}:{encoder.port}")
    print(f"超时时间: {encoder.timeout}s, 最大重试: {encoder.max_retries}")
    print(f"测试文本数量: {len(texts)}")

    try:
        embeddings = encoder.encode(texts, batch_size=encoder.max_batch_size)
        print(f"✅ 请求成功，返回 {len(embeddings)} 条向量，每条维度 {encoder.vector_dim}")
    except Exception as exc:  # noqa: BLE001
        print("❌ 请求失败，捕获到异常：")
        print(f"类型: {type(exc).__name__}")
        print(f"信息: {exc}")
        print("完整堆栈：")
        print(traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_texts = sys.argv[1:]
    else:
        input_texts = [
            "Test sample sentence for embedding connectivity.",
            "第二条测试文本，用于触发批量请求。",
        ]
    run_test(input_texts)

