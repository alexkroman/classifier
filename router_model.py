import torch
from transformers import WhisperForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class WhisperRouter(WhisperForAudioClassification):
    """WhisperForAudioClassification that mean-pools only over real (non-padding) audio frames.

    Whisper pads every clip to 30s; the stock head averages over all 1500 encoder frames, so a 4s clip's
    embedding is ~87% padding. Here the feature extractor's attention mask restricts pooling to the speech.
    """

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_features, output_hidden_states=self.config.use_weighted_layer_sum)
        if self.config.use_weighted_layer_sum:
            stacked = torch.stack(out.hidden_states, dim=1)  # (B, L+1, T, D)
            weights = torch.softmax(self.layer_weights, dim=-1).view(1, -1, 1, 1)
            hidden = (stacked * weights).sum(dim=1)
        else:
            hidden = out.last_hidden_state
        hidden = self.projector(hidden)

        if attention_mask is not None:
            mask = attention_mask[:, ::2].unsqueeze(-1).to(hidden.dtype)  # conv stride 2: 3000 mel -> 1500 frames
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        return SequenceClassifierOutput(logits=self.classifier(pooled))
