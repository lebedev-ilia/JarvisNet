import torch
import torch
import IPython.display as ipd
import torchaudio
from modell import M5
import os


labels = [1, 2, 3]


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint from iteration {}" .format(iteration))
    return model, optimizer, learning_rate, iteration

def index_to_label(index):
    return labels[index]

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def label_to_index(_class):
    return torch.tensor(labels.index(_class))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def inference_model(file_path, tuple_audio, checkpoint_path, num_class):

    labels = [i+1 for i in range(num_class)]

    if file_path != None:
        waveform, sample_rate = torchaudio.load(file_path)
        waveform_first, *_ = torchaudio.load(file_path)
    else:
        waveform, sample_rate = tuple_audio
        waveform_first, *_ = tuple_audio

    ipd.Audio(waveform_first.numpy(), rate=sample_rate)

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)

    print(waveform[0].shape)

    model = M5(n_input=waveform.shape[0], n_output=num_class)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pred = model(transformed.unsqueeze(0))
    pred = get_likely_index(pred)
    pred = index_to_label(pred)

    return pred
