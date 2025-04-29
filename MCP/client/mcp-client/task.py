from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from agentenv.controller import BaseEnvClient

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str
    seq_ids: list[int]
    attention_mask: list[int]
    action_mask: list[int]


TokenizedConversationOutput = TypedDict(
    "TokenizedConversationOutput",
    {
        "text": str,
        "input_ids": list[int],
        "action_mask": list[int],
    },
)


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if message["from"] == "human":
            text = f"<s>[INST] {message['value']} [/INST]"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            text = f"{message['value']}</s>"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            text = f" {text}"
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _tokenize_conversation(
        self,
        conversation: list[ConversationMessage],
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        text = ""
        input_ids = []
        action_mask = []

        for message in conversation:
            message_out = self._tokenize_conversation_one(message, tokenizer)
            text += message_out["text"]
            input_ids += message_out["input_ids"]
            action_mask += message_out["action_mask"]

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _generate_experience_one(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        conversation = list(client.conversation_start)
        conversation.append(
            ConversationMessage({"from": "human", "loss": None, "value": state})
        )
        conversation_tokenized = self._tokenize_conversation(conversation, tokenizer)
        rounds = 0

        while not done:
            input_length = len(conversation_tokenized["input_ids"])
            # if input_length exceeds 4096, break
            if input_length > 4096:
                break
            output = model.generate(
                torch.tensor(
                    [conversation_tokenized["input_ids"]], device=model.device
                ),
                generation_config=generation_config,
            )
            if isinstance(output, GenerateOutput):
                output = output.sequences

            generated_tokens = output[0][input_length:].cpu().numpy().tolist()
            if generated_tokens[-1] != tokenizer.eos_token_id:
                generated_tokens += [tokenizer.eos_token_id]

            generated_text = tokenizer.decode(generated_tokens)
            conversation_tokenized["text"] += f" {generated_text}"
            conversation_tokenized["input_ids"] += generated_tokens
            conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

            generated_text = generated_text[
                : -len(tokenizer.eos_token)
            ]  # not endswith eos_token
            conversation.append(
                ConversationMessage(
                    {"from": "gpt", "loss": True, "value": generated_text}
                )
            )

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            env_message = ConversationMessage(
                {"from": "human", "loss": None, "value": state}
            )
            env_message_tokenized = self._tokenize_conversation_one(
                env_message, tokenizer
            )

            conversation.append(env_message)
            conversation_tokenized["text"] += env_message_tokenized["text"]
            conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
            conversation_tokenized["action_mask"] += env_message_tokenized[
                "action_mask"
            ]

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=conversation_tokenized["text"],
            seq_ids=conversation_tokenized["input_ids"],
            attention_mask=[1] * len(conversation_tokenized["input_ids"]),
            action_mask=conversation_tokenized["action_mask"],
        )

    def _generate_experience_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        # TODO: "Batch experience generation is not implemented. Generate one by one.",
        client = self.clients[0]
        result = [self._generate_experience_one(
                    model=model,
                    tokenizer=tokenizer,
                    client=client,
                    idx=idx,
                    generation_config=generation_config,
                    max_rounds=max_rounds,
                ) for idx in idxs]
        return result

    def generate_experience(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        if isinstance(model, DistributedDataParallel):
            model = model.module

        return self._generate_experience_batch(
            model=model,
            tokenizer=tokenizer,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )

# from dataclasses import dataclass
# from typing import Sequence
# import torch
# import asyncio
# from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
# from mcp import ClientSession

# @dataclass
# class ExperienceOutput:
#     conversation: list[dict]
#     reward: float
#     text: str
#     seq_ids: list[int]
#     attention_mask: list[int]
#     action_mask: list[int]

# class MCPTask:
#     def __init__(
#         self,
#         session: ClientSession,
#         task_type: str,
#         max_context: int = 4096
#     ):
#         self.client = MCPEnvClient(session)
#         self.task_type = task_type
#         self.max_context = max_context

#     def generate(
#         self,
#         model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizerBase,
#         idxs: Sequence[int],
#         gen_config: GenerationConfig,
#         max_rounds: int
#     ) -> list[ExperienceOutput]:
#         """批量生成入口"""
#         return [self._generate_single(model, tokenizer, idx, gen_config, max_rounds) for idx in idxs]

#     def _generate_single(
#         self,
#         model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizerBase,
#         idx: int,
#         gen_config: GenerationConfig,
#         max_rounds: int
#     ) -> ExperienceOutput:
#         """同步方法包装"""
#         return asyncio.run(self._async_generate(idx, model, tokenizer, gen_config, max_rounds))

#     async def _async_generate(
#         self,
#         idx: int,
#         model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizerBase,
#         gen_config: GenerationConfig,
#         max_rounds: int
#     ) -> ExperienceOutput:
#         """异步生成核心逻辑"""
#         # 初始化环境
#         state = await self.client.reset(self.task_type, idx)
#         conversation = [self._build_msg("system", f"Task: {self.task_type}")]
#         conversation.append(self._build_msg("env", state, has_loss=False))
        
#         total_reward = 0.0
#         for _ in range(max_rounds):
#             # 生成响应
#             response = await self._generate_response(
#                 conversation, model, tokenizer, gen_config
#             )
            
#             # 执行动作
#             new_state, reward, done = await self.client.step(
#                 self.task_type, response
#             )
            
#             # 更新对话
#             conversation.append(self._build_msg("assistant", response))
#             conversation.append(self._build_msg("env", new_state, has_loss=False))
            
#             total_reward += reward
#             if done: break

#         return self._build_experience(conversation, tokenizer, total_reward)

#     async def _generate_response(
#         self,
#         conversation: list[dict],
#         model: PreTrainedModel,
#         tokenizer: PreTrainedTokenizerBase,
#         config: GenerationConfig
#     ) -> str:
#         """生成模型响应"""
#         input_ids = self._tokenize_conversation(conversation, tokenizer)
#         if len(input_ids) > self.max_context:
#             input_ids = input_ids[-self.max_context:]
        
#         inputs = torch.tensor([input_ids], device=model.device)
#         outputs = model.generate(inputs, **config.__dict__)
#         return tokenizer.decode(outputs[0][len(input_ids):], skip_special_tokens=True)

#     def _tokenize_conversation(self, conversation: list[dict], tokenizer: PreTrainedTokenizerBase) -> list[int]:
#         """统一编码逻辑"""
#         return tokenizer.encode(
#             "\n".join([msg["value"] for msg in conversation]),
#             add_special_tokens=False
#         )

#     def _build_experience(self, conversation, tokenizer, reward) -> ExperienceOutput:
#         """构建经验输出"""
#         input_ids = self._tokenize_conversation(conversation, tokenizer)
#         return ExperienceOutput(
#             conversation=conversation,
#             reward=reward,
#             text="\n".join([msg["value"] for msg in conversation]),
#             seq_ids=input_ids,
#             attention_mask=[1] * len(input_ids),
#             action_mask=[1 if msg.get("has_loss") else 0 for msg in conversation]
#         )

#     @staticmethod
#     def _build_msg(sender: str, text: str, has_loss: bool = True) -> dict:
#         """统一消息格式"""
#         return {
#             "from": sender,
#             "value": text,
#             "has_loss": has_loss
#         }