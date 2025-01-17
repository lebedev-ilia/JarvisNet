num_classes = 3

class hparams():
  
  batch_size = 64
  log_interval = 1
  n_epoch = 10
  seed = 1234
  lr = 0.001
  wd = 0.0001
  num_classes = 3
  
  cudnn_enabled = False
  cudnn_benchmark = False
  
  distributed_run = False
  
  checkpoint_path = '/Users/user/Desktop/jarvis/JarvisNet/checkpoints/my_model_checkpoint_last.pt'
  output_dir = '/Users/user/Desktop/jarvis/JarvisNet/checkpoints/'
  dataset_dir = '/Users/user/Desktop/jarvis/JarvisNet/jarvis_dataset/'
  
  # scheduler
  sched_step = 500
  sched_gamma = 0.1
  
  
class hparams_colab():
  
  batch_size = 256
  log_interval = 10
  n_epoch = 10
  seed = 1234
  lr = 0.001
  wd = 0.0001
  num_classes = 3
  
  cudnn_enabled = True
  cudnn_benchmark = True
  
  distributed_run = True
  
  checkpoint_path = '/content/'
  output_dir = '/content/'
  dataset_dir = '/content/drive/MyDrive/JarvisDataset'
  
  # scheduler
  sched_step = 20
  sched_gamma = 0.1
  
  