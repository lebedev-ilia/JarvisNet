import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torchaudio
from modell import M34Res
import IPython.display as ipd
from tqdm import tqdm
from dataset_utils import MyDataset
from functions import collate_fn
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
from hparams import hparams, hparams_colab


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt

def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(
        hparams.dist_backend, hparams.dist_url, n_gpus, rank, group_name
    )
    print("Done initializing distributed")
    

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, checkpoint_name):
    filepath = filepath + checkpoint_name
    print("Saving done. Iteration {} to {}".format(iteration, filepath))
    torch.save({
        "iteration": iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate}, filepath)
    

def train(output_dir, dataset_dir, checkpoint_path, colab, n_gpus, rank, group_name, hot_start, hparams):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
        
    torch.manual_seed(hparams.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)
        
    trainset = MyDataset(dataset_dir)

    waveform, sample_rate, _ = trainset[0]

    waveform_first, *_ = trainset[0]
    ipd.Audio(waveform_first.numpy(), rate=sample_rate)
    
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)

    ipd.Audio(transformed.numpy(), rate=new_sample_rate)
    
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
        
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
        
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=hparams.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    model = M34Res(n_input=transformed.shape[0], n_output=hparams.num_classes)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.wd)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hparams.sched_step, gamma=hparams.sched_gamma)  
    
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    transform = transform.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(1, hparams.n_epoch + 1):
    
        train_tqdm = tqdm(train_loader, leave=True)
        
        model.train()
        
        for iteration, (data, target) in enumerate(train_tqdm):
            
            iteration += 1
            
            lr = optimizer.param_groups[0]['lr']

            data = data.to(device)
            target = target.to(device)
            
            data = transform(data)
            output = model(data)
            
            loss = criterion(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            # print training stats
            if iteration % hparams.log_interval == 0:
                train_tqdm.set_description(f"Train Epoch: {epoch} [{iteration * len(data)}/{len(train_loader.dataset)}] Loss: {reduced_loss:.6f}")

            losses.append(loss.item())
            
            scheduler.step()
        
        if epoch == 50:
            checkpoint_name = 'jarvis_checkpoint_{}.pt'.format(epoch)
            
            save_checkpoint(model, optimizer, lr, iteration, output_dir, checkpoint_name)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-d', '--dataset_dir', type=str)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False)
    parser.add_argument('--colab', type=bool, default=False, required=False)
    parser.add_argument('--hot_start', action='store_true', default=False)
    parser.add_argument('--n_gpus', type=int, default=1, required=False, )
    parser.add_argument('--rank', type=int, default=0, required=False)
    parser.add_argument('--group_name', type=str, default='group_name', required=False)

    args = parser.parse_args()
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    if args.colab:
        hparams = hparams_colab()
        train(hparams.output_dir, hparams.dataset_dir, hparams.checkpoint_path, args.colab, args.n_gpus, args.rank, args.group_name, args.hot_start, hparams)
    else:
        hparams = hparams()
        train(hparams.output_dir, hparams.dataset_dir, hparams.checkpoint_path, args.colab, 0, 0, None, args.hot_start, hparams)
