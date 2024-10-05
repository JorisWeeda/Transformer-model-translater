import torch
from torch.utils.data import Dataset


class BiLingualData(Dataset):
    """Dataset class for bilingual text translation with tokenization and padding."""

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_language, target_language, sequence_length):
        """Initializes the dataset with tokenizers and sequence length."""
        super().__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language

        self.sos_token = torch.tensor([source_tokenizer.token_to_id('[SOS]')], dtype=torch.int64, device=self.DEVICE)
        self.eos_token = torch.tensor([source_tokenizer.token_to_id('[EOS]')], dtype=torch.int64, device=self.DEVICE)
        self.pad_token = torch.tensor([source_tokenizer.token_to_id('[PAD]')], dtype=torch.int64, device=self.DEVICE)

        self.sequence_length = sequence_length

    @staticmethod
    def causal_mask(size):
        """Generates a causal mask for self-attention."""
        return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, index):
        """Retrieves a single sample from the dataset."""
        source_target_pair = self.dataset[index]

        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        encoder_num_padding_tokens = self.sequence_length - len(encoder_input_tokens) - 2  # SOS and EOS tokens
        decoder_num_padding_tokens = self.sequence_length - len(decoder_input_tokens) - 1  # SOS token

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError('Given sentence is too long to process.')

        encoder_input = torch.cat([self.sos_token, torch.tensor(encoder_input_tokens, dtype=torch.int64, device=self.DEVICE),
                                   self.eos_token, torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64, device=self.DEVICE)], dim=0)

        decoder_input = torch.cat([self.sos_token, torch.tensor(decoder_input_tokens, dtype=torch.int64, device=self.DEVICE),
                                   torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64, device=self.DEVICE)], dim=0)

        label = torch.cat([torch.tensor(decoder_input_tokens, dtype=torch.int64, device=self.DEVICE),
                           self.eos_token, torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64, device=self.DEVICE)], dim=0)

        assert encoder_input.shape[0] == self.sequence_length
        assert decoder_input.shape[0] == self.sequence_length
        assert label.shape[0] == self.sequence_length

        causal_mask = self.causal_mask(decoder_input.size(0)).to(self.DEVICE)
        
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) & causal_mask

        encoder_mask = encoder_mask.int()
        decoder_mask = decoder_mask.int()

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "source_text": source_text,
            "target_text": target_text,
            "label": label
        }
