from torch import nn
from transformers import BertPreTrainedModel, BertModel


class RerankingModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        labels=None
    ):
        pos_output = self.bert(
            input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos
        )
        pos_dropout = self.dropout(pos_output[1])
        pos_linear = self.classifier(pos_dropout)

        neg_output = self.bert(
            input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg
        )
        neg_dropout = self.dropout(neg_output[1])
        logits_neg = self.classifier(neg_dropout)

        agg = pos_linear - logits_neg
        loss = self.loss_fct(agg.view(-1, self.num_labels), labels.view(-1)) if labels else None

        final_output = (pos_linear,) + pos_output[2:]

        return ((loss,) + final_output) 