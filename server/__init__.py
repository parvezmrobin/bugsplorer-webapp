import numpy as np
import torch
from flask import Flask, request
from flask_cors import CORS
from pympler.asizeof import asizeof

from server.BugPredictionArgs import model_class_of
from server.BugPredictionModel import BugPredictionModel

app = Flask(__name__)
CORS(app)

tokenizer = model_class_of["roberta"].tokenizer.from_pretrained(
    "huggingface/CodeBERTa-small-v1"
)
model_classes = model_class_of["roberta"]
config = model_classes.config.from_pretrained("huggingface/CodeBERTa-small-v1")
python_checkpoint = (
    "./checkpoints/101.train-bugsplorer-defectors-line-random-w_100-gpu_2-b_16"
)
java_checkpoint = "./checkpoints/103.train-bugsplorer-linedp-line-time-w_100-gpu_2-b_16"
print(f"Loading models from {python_checkpoint}, {java_checkpoint}")

device = torch.device("cpu")
MAX_FILE_LEN = 256
MAX_LINE_LEN = 16
python_model, java_model = [
    BugPredictionModel(
        pretrained_model_name=checkpoint,
        config=config,
        encoder_type="line",
        is_checkpoint=True,
        pad_token_id=tokenizer.pad_token_id,
        model_type="roberta",
        max_line_length=MAX_LINE_LEN,
        max_file_length=MAX_FILE_LEN,
        class_weight=torch.tensor(
            [1, 100],
            device=device,
            dtype=torch.float32,
        ),
    )
    for checkpoint in [python_checkpoint, java_checkpoint]
]

python_model.to(device)
python_model.eval()

java_model.to(device)
java_model.eval()

model_parameters = filter(lambda p: p.requires_grad, python_model.parameters())
param_count = sum([np.prod(p.size()) for p in model_parameters])
model_size = asizeof(python_model)
print(
    f"Finished loading two models of size {int(param_count // 1e6)}M params ({model_size / 1e6:.2f}MB)"
)


@app.route("/api/explore", methods=["POST"])
def hello_world():
    lines = request.get_data(as_text=True).replace(tokenizer.eos_token, "").split("\n")
    lang = request.args.get("lang")
    line_token_ids = _tokenize_lines(lines)

    model = python_model
    if lang == "java":
        model = java_model

    loss, logit = model(
        line_token_ids.to(device),
    )
    assert len(logit.shape) == 3
    true_prob = logit[:, :, 1].tolist()  # shape: (num_splits, num_files)
    defect_prob = _get_defect_prob(true_prob)
    return defect_prob[: len(lines)]


def _tokenize_lines(lines) -> torch.Tensor:
    line_token_ids = tokenizer(
        lines,
        max_length=MAX_LINE_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    # split line_token_ids into max_file_len chunks with 64 tokens overlap.
    # this is to avoid truncating lines in the middle of a function call.
    split_line_token_ids = []
    for i in range(0, len(line_token_ids), MAX_FILE_LEN - 64):
        split_line_token_ids.append(line_token_ids[i : i + MAX_FILE_LEN])
    if len(split_line_token_ids) > 1 and len(split_line_token_ids[-1]) <= 64:
        # if the last chunk is less than 64 tokens, remove the last chunk
        # since it is already included in the previous chunk
        split_line_token_ids.pop()

    # pad the last chunk with pad_token_id
    if len(split_line_token_ids[-1]) < MAX_FILE_LEN:
        num_padded_lines = MAX_FILE_LEN - len(split_line_token_ids[-1])
        line_padding = torch.tensor([[tokenizer.pad_token_id]]).repeat(
            num_padded_lines, MAX_LINE_LEN
        )
        split_line_token_ids[-1] = torch.cat(
            [
                split_line_token_ids[-1],
                line_padding,
            ],
            dim=0,
        )

    return torch.stack(split_line_token_ids)


def _get_defect_prob(true_prob: list[list[float]]) -> list[float]:
    """
    :param true_prob: contains the probability of each line in each split being a defect.
    Here, each file is split into multiple splits of 256 lines each with 64 lines overlap.
    :return: Returns the probability of each line being defecting discarding the overlap.
    For each overlap, we take 32 lines from the first split and 32 lines from the second split.
    """

    defect_prob = true_prob[0][:32]
    for i in range(len(true_prob)):
        split = true_prob[i]
        defect_prob.extend(split[32:-32])
    defect_prob.extend(true_prob[-1][:-32])
    return defect_prob
