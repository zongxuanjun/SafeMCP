# from transformers import PreTrainedModel, PreTrainedTokenizerBase


# class Agent:

#     def __init__(
#         self,
#         model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizerBase,
#     ) -> None:
#         self.model = model
#         self.tokenizer = tokenizer


from transformers import PreTrainedModel, PreTrainedTokenizerBase

class Agent:
    """通用智能体封装"""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        session_pool: dict[str, object]  # 多会话支持
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sessions = session_pool

    def detect_tool_call(self, text: str) -> tuple[str, str]:
        """通用工具调用检测"""
        if "[ACTION]" in text and "[/ACTION]" in text:
            start = text.find("[ACTION]") + 8
            end = text.find("[/ACTION]")
            return text[start:end].strip()
        return None