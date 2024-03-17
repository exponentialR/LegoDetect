import torch
import torch.nn as nn
from transformers import ViTModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class ViTObjectDetector(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_decoder_layers=6):
        super(ViTObjectDetector, self).__init__()
        # Load a pre-trained Vision Transformer
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Object query embeddings (number of queries determines the max number of objects you can detect)
        self.obj_queries = nn.Parameter(torch.rand(100, hidden_dim))

        # Linear layers for bounding box prediction and class prediction
        self.bbox_head = nn.Linear(hidden_dim, 4)  # Predicting 4 values for bbox (x, y, width, height)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # num_classes + 1 for background

        # Positional embeddings for the transformer decoder
        self.positional_embeddings = nn.Parameter(torch.rand(100, hidden_dim))

    def forward(self, pixel_values):
        # Backbone ViT feature extraction
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Expand object queries and positional embeddings to match batch size
        batch_size, seq_len, hidden_dim = last_hidden_state.shape
        obj_queries = self.obj_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_queries, hidden_dim)
        pos_embeddings = self.positional_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        # Decoder to predict object bboxes and classes
        decoder_output = self.transformer_decoder(obj_queries, last_hidden_state, pos=pos_embeddings)

        # Prediction heads
        bbox_outputs = self.bbox_head(decoder_output)
        class_outputs = self.class_head(decoder_output)

        return {'pred_logits': class_outputs, 'pred_boxes': bbox_outputs}

if __name__ == '__main__':
    num_classes = 12  # Including background as a class
    model = ViTObjectDetector(num_classes=num_classes)
    pixel_values = torch.rand(1, 3, 224, 224)  # Batch size of 1

    outputs = model(pixel_values)
    print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)

