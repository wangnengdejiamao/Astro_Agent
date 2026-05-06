from __future__ import annotations

import pickle
import socket
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from utils.logger import logger


class EmbeddingService:
    """
    与外部嵌入服务交互并提供向量缓存的工具类。
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int,
        max_retries: int,
        batch_size: int,
        max_workers: int,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_workers = max_workers

        self._cache: Dict[str, np.ndarray] = {}
        self._tested = False
        self._available = True

    @property
    def available(self) -> bool:
        if not self._tested:
            self._available = self._test_connection()
        return self._available

    def clear_cache(self):
        self._cache.clear()

    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        if not self.available:
            return None

        try:
            embeddings = self._request_embeddings([text])
            if embeddings and embeddings[0]:
                vector = np.array(embeddings[0], dtype=np.float32)
                self._cache[text] = vector
                return vector
            logger.error("计算 embedding 失败: 返回结果为空")
        except Exception as exc:  # pragma: no cover - 网络异常环境
            logger.error("计算 embedding 失败: %s", exc)
        return None

    def compute_embeddings_batch(self, texts: Iterable[str]) -> List[Optional[np.ndarray]]:
        texts_list = list(texts)
        if not texts_list:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(texts_list)
        missing = []

        for idx, text in enumerate(texts_list):
            if text in self._cache:
                results[idx] = self._cache[text]
            else:
                missing.append((idx, text))

        if not missing or not self.available:
            return results

        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            indices, payload = zip(*batch)
            try:
                embeddings = self._request_embeddings(list(payload))
            except Exception as exc:  # pragma: no cover
                logger.error("批量计算 embedding 失败: %s", exc)
                embeddings = []

            for idx, embedding in zip(indices, embeddings):
                if embedding:
                    vector = np.array(embedding, dtype=np.float32)
                    text = texts_list[idx]
                    self._cache[text] = vector
                    results[idx] = vector

        return results

    def _test_connection(self) -> bool:
        if self._tested:
            return self._available

        try:
            result = self._request_embeddings(["ping"])
            self._available = bool(result and len(result) == 1 and len(result[0]) > 0)
            if self._available:
                logger.info("Embedding 服务连接成功: %s:%s", self.host, self.port)
        except Exception as exc:  # pragma: no cover
            logger.error("Embedding 服务连接失败: %s", exc)
            self._available = False
        finally:
            self._tested = True
        return self._available

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        for attempt in range(self.max_retries):
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(self.timeout)
                client.connect((self.host, self.port))

                payload = pickle.dumps(texts)
                client.send(len(payload).to_bytes(4, byteorder="big"))
                client.sendall(payload)

                header = client.recv(4)
                if len(header) != 4:
                    raise RuntimeError("接收数据长度失败")

                body_len = int.from_bytes(header, byteorder="big")
                body = b""
                while len(body) < body_len:
                    chunk = client.recv(min(4096, body_len - len(body)))
                    if not chunk:
                        raise RuntimeError("连接意外断开")
                    body += chunk

                client.close()
                return pickle.loads(body)
            except Exception as exc:  # pragma: no cover
                logger.error("Embedding 服务调用失败: %s (第 %d 次)", exc, attempt + 1)
                if "client" in locals():
                    try:
                        client.close()
                    except Exception:
                        pass
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return []


__all__ = ["EmbeddingService"]

