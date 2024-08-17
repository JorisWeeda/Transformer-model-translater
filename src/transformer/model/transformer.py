import torch.nn as nn
from src.transformer.layers import InputEmbedding, PositionalEncoding, MultiheadAttention, FeedForward, ProjectionLayer
from src.transformer.blocks import EncoderBlock, DecoderBlock
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    """Transformer model with encoder and decoder."""

    DEFAULT_D_MODEL = 512
    DEFAULT_DROPOUT = 0.1
    DEFAULT_D_FF = 2048
    DEFAULT_N = 6
    DEFAULT_H = 8

    def __init__(self, encoder, decoder, encoder_embedding, decoder_embedding, encoder_position, decoder_position, projection_layer):
        """Initializes the Transformer model."""
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding

        self.encoder_position = encoder_position
        self.decoder_position = decoder_position

        self.projection_layer = projection_layer


    @classmethod
    def build_transformer(cls, source_vocabulary_size, target_vocabulary_size, source_sequence_len, target_sequence_len,
                          d_model=DEFAULT_D_MODEL, N=DEFAULT_N, h=DEFAULT_H, dropout=DEFAULT_DROPOUT, d_ff=DEFAULT_D_FF):
        """Builds and initializes a Transformer model."""
        encoder_embedding = InputEmbedding(d_model, source_vocabulary_size)
        decoder_embedding = InputEmbedding(d_model, target_vocabulary_size)

        encoder_position = PositionalEncoding(d_model, source_sequence_len, dropout)
        decoder_position = PositionalEncoding(d_model, target_sequence_len, dropout)

        encoder_blocks = []
        for _ in range(N):
            self_attention = MultiheadAttention(d_model, h, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout)
            encoder_blocks.append(EncoderBlock(self_attention, feed_forward, dropout))

        decoder_blocks = []
        for _ in range(N):
            self_attention = MultiheadAttention(d_model, h, dropout)
            cross_attention = MultiheadAttention(d_model, h, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout)
            decoder_blocks.append(DecoderBlock(self_attention, cross_attention, feed_forward, dropout))

        projection_layer = ProjectionLayer(d_model, target_vocabulary_size)

        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        transformer = cls(encoder, decoder, encoder_embedding, decoder_embedding, encoder_position, decoder_position, projection_layer)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer

    def encode(self, source, encoder_mask):
        """Encodes the source sequence."""
        source = self.encoder_embedding(source)
        source = self.encoder_position(source)
        return self.encoder(source, encoder_mask)

    def decode(self, encoder_output, encoder_mask, target, decoder_mask):
        """Decodes the target sequence."""
        target = self.decoder_embedding(target)
        target = self.decoder_position(target)
        return self.decoder(target, encoder_output, encoder_mask, decoder_mask)

    def projection(self, x):
        """Applies the projection layer."""
        return self.projection_layer(x)
