from typing import NamedTuple, Type, Optional

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaTokenizerFast,
    T5Config,
    RobertaModel,
    T5EncoderModel,
)

from server.FileEncoders import (
    RobertaForLineClassification,
    BertForLineClassification,
    T5ForLineClassification,
)


class ModelClasses(NamedTuple):
    config: Type[PretrainedConfig]
    line_encoder: Type[PreTrainedModel]
    file_encoder: Type[PreTrainedModel]
    tokenizer: Type[PreTrainedTokenizerFast]


model_class_of: dict[str, ModelClasses] = {
    "roberta": ModelClasses(
        RobertaConfig, RobertaModel, RobertaForLineClassification, RobertaTokenizerFast
    ),
    "bert": ModelClasses(
        RobertaConfig, RobertaModel, BertForLineClassification, RobertaTokenizerFast
    ),
    "t5": ModelClasses(
        T5Config,
        T5EncoderModel,
        T5ForLineClassification,
        RobertaTokenizerFast,
    ),
}
