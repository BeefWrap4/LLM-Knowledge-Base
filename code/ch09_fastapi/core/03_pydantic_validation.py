# ---
# chapter: 9
# topic: Web开发与FastAPI
# section: 9.3.2 Pydantic 模型验证
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: pydantic
# run: python 03_pydantic_validation.py
# expected_runtime: <1s
# expected_output: 正常请求通过校验 + 触发跨字段校验
# ---
# See: ../tutorial/09_Web开发与FastAPI.md (lines 222-258)
# Interview hooks:
#   1. field_validator 与 model_validator(mode="after") 的执行顺序与适用场景？
#   2. Pydantic v2 中 Literal 字段如何生成 OpenAPI 的 enum 约束？
#   3. model_dump() 与 dict() 有什么区别？何时使用 model_dump_json()？
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingRequest(BaseModel):
    """Embedding 请求模型 - 展示高级验证"""

    texts: list[str] = Field(min_length=1, max_length=100, description="待编码文本列表")
    model: Literal["text-embedding-3-small", "text-embedding-3-large", "bge-m3"] = Field(
        default="bge-m3", description="Embedding 模型"
    )
    normalize: bool = Field(default=True, description="是否归一化")

    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, texts: list[str]) -> list[str]:
        """字段级验证器：确保每个文本非空"""
        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"第 {i} 个文本不能为空字符串")
        return texts

    @model_validator(mode="after")
    def check_model_compatibility(self):
        """模型级验证器：检查跨字段一致性"""
        if self.model == "text-embedding-3-small" and len(self.texts) > 50:
            raise ValueError("small 模型单次最多处理 50 条文本")
        return self


if __name__ == "__main__":
    # 使用示例
    valid_req = EmbeddingRequest(texts=["FastAPI 教程", "机器学习基础"], model="bge-m3", normalize=True)
    print(valid_req.model_dump())  # 序列化为字典
    print(valid_req.model_dump_json())  # 序列化为 JSON 字符串

    # 字段级校验：空字符串文本
    try:
        EmbeddingRequest(texts=["ok", "  ", "fine"])
    except Exception as e:
        print(f"[字段级校验触发] {type(e).__name__}: 不能为空")

    # 模型级校验：small 模型 + 超过 50 条
    try:
        EmbeddingRequest(texts=[f"text-{i}" for i in range(51)], model="text-embedding-3-small")
    except Exception as e:
        print(f"[模型级校验触发] {type(e).__name__}: small 模型单次最多处理 50 条文本")
