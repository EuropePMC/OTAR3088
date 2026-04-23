"""
Implements a Conditonal Random Field(CRF)
head with a Bert-Like Model
"""

from loguru import logger
import torch
import torch.nn as nn
from torchcrf import CRF

from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput


class BERTCRFForTokenClassification(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "bert"
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.crf = CRF(self.num_labels, batch_first=True)
        #self.bert = AutoModel.from_pretrained(config._name_or_path, config)
        #self.bert = None
        self.bert = AutoModel.from_config(config)

        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

        

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                labels=None, 
                **kwargs):

        outputs = self.bert(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                **kwargs)


        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        emissions = self.classifier(sequence_output)
       

        loss = None
        if labels is not None:
            mask = (attention_mask.bool()) & (labels != -100)
            mask = attention_mask.bool()
            #mask[:,0] = True
            valid_labels = labels.clone()
            valid_labels[valid_labels == -100] = 0

            loss = -self.crf(emissions, 
                             valid_labels, 
                              mask=mask, 
                              reduction='mean')



        return TokenClassifierOutput(
                            loss=loss,
                            logits=emissions, 
                            hidden_states=outputs.hidden_states,
                            attentions=outputs.attentions,
                            )
