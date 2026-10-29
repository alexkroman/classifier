import torch
from torch import nn
from transformers import MoonshineModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.moonshine.modeling_moonshine import MoonshineEncoder, MoonshinePreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

MOONSHINE_STRIDE = 64 * 3 * 2  # Moonshine's conv strides: raw samples per encoder frame


def masked_mean(hidden, mask):
    """(B, T, P) -> (B, P), averaging over real (non-padding) frames only."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


class MoonshineRouter(MoonshinePreTrainedModel):
    """Audio classifier on Moonshine's encoder: raw 16 kHz waveform in, variable length (no 30s padding).

    A learned softmax-weighted sum of the encoder's layers (conv stem + every layer), a per-frame
    Linear(hidden_size -> config.classifier_proj_size), a masked mean over real frames, then an optional GELU layer
    of width config.head_hidden (0 / absent = linear head) and the classifier. Without the GELU layer,
    projector -> pool -> classifier is one linear map. Build a fresh one from the moonshine-tiny weights with
    from_moonshine(); save_pretrained / from_pretrained round-trip as usual.

    config.specaug_time / specaug_feat: SpecAugment for fine-tuning (see _spec_augment); 0 / absent = off.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = MoonshineEncoder(config)
        n = config.encoder_num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.ones(n) / n)
        p = config.classifier_proj_size
        self.projector = nn.Linear(config.hidden_size, p)
        hidden = getattr(config, "head_hidden", 0)
        self.head_hidden = nn.Sequential(nn.Linear(p, hidden), nn.GELU()) if hidden else nn.Identity()
        self.classifier = nn.Linear(hidden or p, config.num_labels)
        self.post_init()
        self.encoder.conv3.register_forward_hook(self._spec_augment)

    def _spec_augment(self, module, inputs, out):
        """SpecAugment for a raw-waveform encoder, as in wav2vec2 fine-tuning: while training, zero random time spans
        (10 frames, ~0.24s) and channel bands (16 of 288) of the conv front end's output, before the transformer.
        Zeroing conv3's pre-activation output zeroes the features (gelu(0) = 0). The config values are the fraction
        of frames / channels masked; read at call time, so they can be set on a loaded model."""
        p_time, p_feat = getattr(self.config, "specaug_time", 0.0), getattr(self.config, "specaug_feat", 0.0)
        if not self.training or not (p_time or p_feat):
            return out
        batch, channels, frames = out.shape
        if p_time:
            m = torch.from_numpy(_compute_mask_indices((batch, frames), p_time, 10, min_masks=1)).to(out.device)
            out = out.masked_fill(m[:, None, :], 0.0)
        if p_feat:
            m = torch.from_numpy(_compute_mask_indices((batch, channels), p_feat, 16, min_masks=1)).to(out.device)
            out = out.masked_fill(m[:, :, None], 0.0)
        return out

    @classmethod
    def from_moonshine(cls, base, config):
        """New router whose encoder is copied from a pretrained Moonshine ASR checkpoint; the head is fresh."""
        model = cls(config)
        model.encoder.load_state_dict(MoonshineModel.from_pretrained(base).encoder.state_dict())
        return model

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_values, attention_mask=attention_mask, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states, dim=1)  # (B, L+1, T, D)
        weights = torch.softmax(self.layer_weights, dim=-1).view(1, -1, 1, 1)
        hidden = self.projector((stacked * weights).sum(dim=1))  # (B, T, P)

        if attention_mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:  # the same subsampling the encoder applies to its own attention mask
            mask = attention_mask[:, ::MOONSHINE_STRIDE][:, : hidden.shape[1]].bool()
        pooled = masked_mean(hidden, mask)
        return SequenceClassifierOutput(logits=self.classifier(self.head_hidden(pooled)))


def load_router(path):
    return MoonshineRouter.from_pretrained(path)
