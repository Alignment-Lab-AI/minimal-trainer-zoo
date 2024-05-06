# Training a custom image classification model with the trainer API
# docs : https://huggingface.co/docs/transformers/en/main_classes/trainer
# adapted from : https://colab.research.google.com/drive/1u9r-p_x7QXH9zAbQ5c0O2smEBHvC44me?usp=sharing
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Image, load_dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments

# Constants
accuracy = evaluate.load("accuracy")
dataset = load_dataset("mnist")

# Processing the dataset
dataset = dataset.cast_column("image", Image(mode="RGB"))
transform = transforms.Compose([transforms.ToTensor()])


def to_pt(batch):
    batch["pixel_values"] = [transform(image.convert("RGB")) for image in batch["image"]]
    return batch


train = dataset["train"].with_transform(to_pt)
test = dataset["test"].with_transform(to_pt)


# Creating the model
class BasicNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pixel_values, labels=None):
        # the labels parameter allows us to finetune our model
        # with the Trainer API easily
        x = F.relu(F.max_pool2d(self.conv1(pixel_values), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        logits = self.softmax(x)
        if labels is not None:
            # this will make your AI compatible with the trainer API
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits


# Initialize the model
model = BasicNet(channels=3)  # RGB


# Define the metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Create a data collator
def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


# Setup the training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_mnist_model",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    logging_steps=100,
    push_to_hub=True,
)

# configure the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_metrics,
)
# train the model
trainer.train()
