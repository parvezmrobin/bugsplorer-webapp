import numpy as np
import torch
from flask import Flask

from server.BugPredictionArgs import model_class_of
from server.BugPredictionModel import BugPredictionModel

app = Flask(__name__)

tokenizer = model_class_of['roberta'].tokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')
model_classes = model_class_of['roberta']
config = model_classes.config.from_pretrained('huggingface/CodeBERTa-small-v1')
checkpoint = './checkpoints/101.train-bugsplorer-defectors-line-random-w_100-gpu_2-b_16'
print(f"Loading 'roberta' model from {checkpoint}")

device = torch.device('cpu')
model = BugPredictionModel(
    pretrained_model_name=checkpoint,
    config=config,
    encoder_type='line',
    is_checkpoint=True,
    pad_token_id=tokenizer.pad_token_id,
    model_type='roberta',
    max_line_length=16,
    max_file_length=512,
    class_weight=torch.tensor(
        [1, 100],
        device=device,
        dtype=torch.float32,
    ),
)
model.to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_size = sum([np.prod(p.size()) for p in model_parameters])
print(
    f"Finished loading model of size {int(model_size // 1e6)}M"
)


@app.route("/api/explore")
def hello_world():
    return "<p>Hello, World!</p>"
