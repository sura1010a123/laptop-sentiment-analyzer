import torch
import torch.nn as nn
from transformers import BertModel


class AspectSentimentModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_classes=3, dropout=0.3):
        """
        num_classes:
        0 -> negative
        1 -> neutral
        2 -> positive
        """
        super(AspectSentimentModel, self).__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(pretrained_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected classification layer
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: token ids from tokenizer
        attention_mask: mask for padding tokens
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token output
        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output)

        logits = self.fc(x)

        return logits
    