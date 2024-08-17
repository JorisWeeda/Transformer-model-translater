
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import BiLingualData
from src.transformer.model import Transformer


TRAIN_SPLIT_PERCENTAGE = 0.9
EPSILON = 10e-6


def get_weights_file_path(config, epoch):
    """Generate the path for the weights file."""
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)

        tokenizer.train_from_iterator(
            get_all_sentences(dataset, language), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config["source_language"]}-{config["target_language"]}', split='train')

    len_training_data = int(TRAIN_SPLIT_PERCENTAGE * len(dataset_raw))

    source_tokenizer = get_or_build_tokenizer(
        config, dataset_raw, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(
        config, dataset_raw, config['target_language'])

    train_data_raw, validation_data_raw = random_split(
        dataset_raw, [len_training_data, len(dataset_raw) - len_training_data])

    train_data = BiLingualData(train_data_raw, source_tokenizer, target_tokenizer,
                               config["source_language"], config["target_language"], config["sequence_length"])

    validation_data = BiLingualData(validation_data_raw, source_tokenizer, target_tokenizer,
                                    config["source_language"], config["target_language"], config["sequence_length"])

    max_source_sentence_len = 0
    max_target_sentence_len = 0

    for item in dataset_raw:
        source_ids = source_tokenizer.encode(
            item['translation'][config["source_language"]]).ids
        target_ids = target_tokenizer.encode(
            item['translation'][config["target_language"]]).ids

        max_source_sentence_len = max(max_source_sentence_len, len(source_ids))
        max_target_sentence_len = max(max_target_sentence_len, len(target_ids))

    print(f'max_source_sentence_len: {max_source_sentence_len}')
    print(f'max_target_sentence_len: {max_target_sentence_len}')

    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(
        validation_data, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer


def get_model(config, source_vocabulary_lengh, target_vocabulary_lengh):
    model = Transformer.build_transformer(source_vocabulary_lengh, target_vocabulary_lengh,
                              config["sequence_length"], config["sequence_length"],
                              config["d_model"])
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Logger | Using device {device}.')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size())

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], eps=EPSILON)

    initial_epoch = 0
    global_step = 0


    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Logger | Preloading model {model_filename}.')

        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])

        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)

            projection_output = model.projection_layer(decoder_output)
            label = batch['label'].to(device)

            loss = loss_fn(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({'loss' f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        model_save_data = {'epoch': epoch, 'model_state_dict': model.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict(), global_step: global_step}

        torch.save(model_save_data, model_filename)


