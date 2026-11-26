





HOOPS

- hoops ai官网

  - https://docs.techsoft3d.com/hoops/ai/

- hoops ai license key：

  - ~~~
    4Ezn6gR24yu2AEMH0CnpDeNyBEUz2CU7vUA5AQMz9BHEjr9Elk9Mj79EjrB68U779U5FjDVc4eIz7QQL4TV54E36BguRxU388UR69yyIxDa83yMJ1QQ1xAIM4E6K1jeM9QZc0jeG4Ba8CgeM4Urk3i6P1hfy9TJd0TMM0U60DRiUwgNr0VeYwhf6DUzm1AYSwBVpDCi38fJc8gR2CBQLxgV31DR60QNzEfUWC79EjrACj7DFrbDFj7DFj7DFj7DFIspFjAdHj7Eij7DFj7DFjGtPj7DFj7DFIspFjHmPIqQJwReJ1TNkBSa7wTYH2g2NBfjtCyY2xxnx1vnbxRNz9hI37QU33CZswhf8Ehj90AZnDzaKwuMV1U6KBxby6faW5sQYzyYgj7DFGrHFjFqNEinsAyfxwwv1DyJk9i6J9iILAuET2ja2AfYMwyztCUmM1SnkxCmG5gR99xfdvDR$2iNzACz13SQXzUbqwvnuCTN20vaOAiV$1yYW2wYX5SntEhrb1EeJ8FqN5CBvAEqI3TJy3fZ2xhq4wBMUAEAN4Tnw3Se3CwIVDDQ01VrawS60AyAGCifb8wr23xUSCjn5AwFr6fe4vAFlDFi5AC30AiV7Axf1Bg230Sn43Bjo3TU_0eIM3gzqBhQN0Tn91jnq3hf30QZ59iFbvViYBfiT5EfkvE3zxCYI3eR6BRJmDhjczRiR1DrnxSaT2DN$CCN$AS3bDvQ7BuJtwiiN6eEVEg20BiQOwfnv8gTFj7DFIspFj7DFj7GhjNDF7QNlxRaz7Rfpxfjmwya68QY1xuFo8xRowBe3x7Foxfnn8vjlxuVk8QY7vTI08DNnxRboweQzxRa77TU2xBe1xRazvRi6wve78ybnvDQ78fnpwxJk8uE3xTVp9eQ4xAM3wfboxDZkweNmvBa3wDNowxVmwxQ3xAE38AM28eY6vDZlxuY8weVnwxVovDM58TZovTZmxDRk8hVo9fm88QNmwRnk8uU58uM07TQ5wfe18QI1wAU8wvi6vUa27TY7xfi6xvi58DY28QFlweE4xeY37QMzvTJmwuFnwUa0xhRk8TU88TI7vQU6xhI18ve4wAVpxDI1whVo9hY48vbk7QM18fa08Ea1wfe8xuI4xTQ0xuZmvRjm7TY6xRi2wRe38BfoxDRlxxVlwBjl9fi4wxVlxQJl8eNpwve6wAQ1vQY38hVkwAU7vDM8xuZl8AJnxQNp8Rnk8eJm8eFp8eFm8fbm8ya88xY18RfkxRaz7TZnxybnxAQ7xhJmvBblwQQ1wAQ8xAZnwAZl8xVpxTU2vQU88AZnwDY8xuU4xTI48Ebpx7ES7UN7
    ~~~

- <img src="/Users/liangsiyu/Library/Application Support/typora-user-images/image-20260706084246862.png" alt="image-20260706084246862" style="zoom:30%;" />

- installation (参考hoops ai官网装环境)

  - ~~~
    apt-get update
    apt-get install -y libglu1-mesa
    apt-get install -y xvfb
    
    export LIBGL_ALWAYS_SOFTWARE=1
    ~~~

- Run 

  - ~~~
    xvfb-run -a python your_script_name.py
    ~~~

- train.py

  - 是根据HOOPS官网训练脚本写的，训练分类的模型
  - 训练HOOPS的embedding模型参考：
    - https://docs.techsoft3d.com/hoops/ai/programming_guide/shape-embeddings.html






### models  architecture 

- ~~~
  Found 4 checkpoint files:
    ts3d_162k_mfr.ckpt (62.08 MB)
    ts3d_1M_hoops_embeddings.ckpt (14.62 MB)
    ts3d_1M_hoops_embeddings.pt (11.58 MB)
    ts3d_graphclassification_5k_10epochs.ckpt (15.38 MB)
  
  ================================================================================
  File: ts3d_162k_mfr.ckpt (62.08 MB)
  ================================================================================
  
  📊 Checkpoint loaded successfully!
  Top-level keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
  
  ⚡ PyTorch Lightning Checkpoint
  Epoch: 52
  Global step: 23532
  
  🔧 Hyperparameters:
    args: NoneType
    num_classes: 25
    n_layers_encode: 8
    dim_node: 256
    d_model: 512
    n_heads: 32
    dropout: 0.3
    attention_dropout: 0.3
    act_dropout: 0.3
    result_dir: PurePosixPath
  
  🏗️ Model Architecture:
  Total layers: 288
  Total parameters: 5,404,914
  
  📦 Module Groups (3 total):
  
    attention (2 layers):
      attention.dense_weight.weight: [1, 256] (torch.float32)
      attention.dense_weight.bias: [1] (torch.float32)
  
    brep_encoder (266 layers):
      brep_encoder.graph_node_feature.surf_encoder.conv1.0.weight: [64, 7, 3, 3] (torch.float32)
      brep_encoder.graph_node_feature.surf_encoder.conv1.1.weight: [64] (torch.float32)
      brep_encoder.graph_node_feature.surf_encoder.conv1.1.bias: [64] (torch.float32)
      brep_encoder.graph_node_feature.surf_encoder.conv1.1.running_mean: [64] (torch.float32)
      brep_encoder.graph_node_feature.surf_encoder.conv1.1.running_var: [64] (torch.float32)
  
    classifier (20 layers):
      classifier.linear1.weight: [512, 256] (torch.float32)
      classifier.bn1.weight: [512] (torch.float32)
      classifier.bn1.bias: [512] (torch.float32)
      classifier.bn1.running_mean: [512] (torch.float32)
      classifier.bn1.running_var: [512] (torch.float32)
  
  📝 Layer Types:
    Tensor: 288
  
  🔍 First 30 layers:
    brep_encoder.graph_node_feature.surf_encoder.conv1.0.weight: [64, 7, 3, 3], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv1.1.weight: [64], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv1.1.bias: [64], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv1.1.running_mean: [64], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv1.1.running_var: [64], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv1.1.num_batches_tracked: [], dtype=torch.int64
    brep_encoder.graph_node_feature.surf_encoder.conv2.0.weight: [128, 64, 3, 3], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv2.1.weight: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv2.1.bias: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv2.1.running_mean: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv2.1.running_var: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv2.1.num_batches_tracked: [], dtype=torch.int64
    brep_encoder.graph_node_feature.surf_encoder.conv3.0.weight: [256, 128, 3, 3], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv3.1.weight: [256], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv3.1.bias: [256], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv3.1.running_mean: [256], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv3.1.running_var: [256], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.conv3.1.num_batches_tracked: [], dtype=torch.int64
    brep_encoder.graph_node_feature.surf_encoder.fc.0.weight: [128, 256], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.fc.1.weight: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.fc.1.bias: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.fc.1.running_mean: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.fc.1.running_var: [128], dtype=torch.float32
    brep_encoder.graph_node_feature.surf_encoder.fc.1.num_batches_tracked: [], dtype=torch.int64
    brep_encoder.graph_node_feature.face_area_encoder.linear1.weight: [32, 1], dtype=torch.float32
    brep_encoder.graph_node_feature.face_area_encoder.bn1.weight: [32], dtype=torch.float32
    brep_encoder.graph_node_feature.face_area_encoder.bn1.bias: [32], dtype=torch.float32
    brep_encoder.graph_node_feature.face_area_encoder.bn1.running_mean: [32], dtype=torch.float32
    brep_encoder.graph_node_feature.face_area_encoder.bn1.running_var: [32], dtype=torch.float32
    brep_encoder.graph_node_feature.face_area_encoder.bn1.num_batches_tracked: [], dtype=torch.int64
  
  ================================================================================
  File: ts3d_1M_hoops_embeddings.ckpt (14.62 MB)
  ================================================================================
  
  📊 Checkpoint loaded successfully!
  Top-level keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
  
  ⚡ PyTorch Lightning Checkpoint
  Epoch: 8
  Global step: 50301
  
  🔧 Hyperparameters:
    lr: 0.0003
    weight_decay: 0.0
    emb_dim: 1024
    proj_dim: 512
    surf_in_ch: 23
    curve_in_ch: 6
    proj_hidden: 1024
    use_bn: True
    temp_init: 0.05
    temp_min: 0.01
    temp_max: 0.2
    surf_c: tuple
    curve_c: tuple
    conv_kernel: 3
    conv_pad: 1
    aug_noise_std: 0.005
    aug_scale_min: 0.85
    aug_scale_max: 1.2
    p_hflip: 0.5
    p_vflip: 0.5
  
  🏗️ Model Architecture:
  Total layers: 65
  Total parameters: 3,028,715
  
  📦 Module Groups (1 total):
  
    model (65 layers):
      model.encoder_surface.stem.0.0.weight: [32, 23, 3, 3] (torch.float32)
      model.encoder_surface.stem.0.0.bias: [32] (torch.float32)
      model.encoder_surface.stem.0.1.weight: [32] (torch.float32)
      model.encoder_surface.stem.0.1.bias: [32] (torch.float32)
      model.encoder_surface.stem.0.1.running_mean: [32] (torch.float32)
  
  📝 Layer Types:
    Tensor: 65
  
  🔍 First 30 layers:
    model.encoder_surface.stem.0.0.weight: [32, 23, 3, 3], dtype=torch.float32
    model.encoder_surface.stem.0.0.bias: [32], dtype=torch.float32
    model.encoder_surface.stem.0.1.weight: [32], dtype=torch.float32
    model.encoder_surface.stem.0.1.bias: [32], dtype=torch.float32
    model.encoder_surface.stem.0.1.running_mean: [32], dtype=torch.float32
    model.encoder_surface.stem.0.1.running_var: [32], dtype=torch.float32
    model.encoder_surface.stem.0.1.num_batches_tracked: [], dtype=torch.int64
    model.encoder_surface.stem.1.0.weight: [64, 32, 3, 3], dtype=torch.float32
    model.encoder_surface.stem.1.0.bias: [64], dtype=torch.float32
    model.encoder_surface.stem.1.1.weight: [64], dtype=torch.float32
    model.encoder_surface.stem.1.1.bias: [64], dtype=torch.float32
    model.encoder_surface.stem.1.1.running_mean: [64], dtype=torch.float32
    model.encoder_surface.stem.1.1.running_var: [64], dtype=torch.float32
    model.encoder_surface.stem.1.1.num_batches_tracked: [], dtype=torch.int64
    model.encoder_surface.stem.2.0.weight: [128, 64, 3, 3], dtype=torch.float32
    model.encoder_surface.stem.2.0.bias: [128], dtype=torch.float32
    model.encoder_surface.stem.2.1.weight: [128], dtype=torch.float32
    model.encoder_surface.stem.2.1.bias: [128], dtype=torch.float32
    model.encoder_surface.stem.2.1.running_mean: [128], dtype=torch.float32
    model.encoder_surface.stem.2.1.running_var: [128], dtype=torch.float32
    model.encoder_surface.stem.2.1.num_batches_tracked: [], dtype=torch.int64
    model.encoder_surface.stem.3.log_p: [], dtype=torch.float32
    model.encoder_surface.fc.weight: [1024, 128], dtype=torch.float32
    model.encoder_surface.fc.bias: [1024], dtype=torch.float32
    model.encoder_curve.stem.0.0.weight: [32, 6, 3], dtype=torch.float32
    model.encoder_curve.stem.0.0.bias: [32], dtype=torch.float32
    model.encoder_curve.stem.0.1.weight: [32], dtype=torch.float32
    model.encoder_curve.stem.0.1.bias: [32], dtype=torch.float32
    model.encoder_curve.stem.0.1.running_mean: [32], dtype=torch.float32
    model.encoder_curve.stem.0.1.running_var: [32], dtype=torch.float32
  
  ================================================================================
  File: ts3d_1M_hoops_embeddings.pt (11.58 MB)
  ================================================================================
  
  📊 Checkpoint loaded successfully!
  Top-level keys: ['model_state_dict', 'epoch', 'val_hard_acc']
  
  🔥 PyTorch Model Checkpoint
  
  🏗️ Model Architecture:
  Total layers: 65
  Total parameters: 3,028,715
  
  📦 Module Groups (4 total):
  
    encoder_curve (24 layers):
      encoder_curve.stem.0.0.weight: [32, 6, 3] (torch.float32)
      encoder_curve.stem.0.0.bias: [32] (torch.float32)
      encoder_curve.stem.0.1.weight: [32] (torch.float32)
      encoder_curve.stem.0.1.bias: [32] (torch.float32)
      encoder_curve.stem.0.1.running_mean: [32] (torch.float32)
  
    encoder_surface (24 layers):
      encoder_surface.stem.0.0.weight: [32, 23, 3, 3] (torch.float32)
      encoder_surface.stem.0.0.bias: [32] (torch.float32)
      encoder_surface.stem.0.1.weight: [32] (torch.float32)
      encoder_surface.stem.0.1.bias: [32] (torch.float32)
      encoder_surface.stem.0.1.running_mean: [32] (torch.float32)
  
    proj_head (16 layers):
      proj_head.fc1.weight: [1024, 1024] (torch.float32)
      proj_head.fc1.bias: [1024] (torch.float32)
      proj_head.bn1.weight: [1024] (torch.float32)
      proj_head.bn1.bias: [1024] (torch.float32)
      proj_head.bn1.running_mean: [1024] (torch.float32)
  
    temperature (1 layers):
      temperature.u: [] (torch.float32)
  
  📝 Layer Types:
    Tensor: 65
  
  🔍 First 30 layers:
    encoder_surface.stem.0.0.weight: [32, 23, 3, 3], dtype=torch.float32
    encoder_surface.stem.0.0.bias: [32], dtype=torch.float32
    encoder_surface.stem.0.1.weight: [32], dtype=torch.float32
    encoder_surface.stem.0.1.bias: [32], dtype=torch.float32
    encoder_surface.stem.0.1.running_mean: [32], dtype=torch.float32
    encoder_surface.stem.0.1.running_var: [32], dtype=torch.float32
    encoder_surface.stem.0.1.num_batches_tracked: [], dtype=torch.int64
    encoder_surface.stem.1.0.weight: [64, 32, 3, 3], dtype=torch.float32
    encoder_surface.stem.1.0.bias: [64], dtype=torch.float32
    encoder_surface.stem.1.1.weight: [64], dtype=torch.float32
    encoder_surface.stem.1.1.bias: [64], dtype=torch.float32
    encoder_surface.stem.1.1.running_mean: [64], dtype=torch.float32
    encoder_surface.stem.1.1.running_var: [64], dtype=torch.float32
    encoder_surface.stem.1.1.num_batches_tracked: [], dtype=torch.int64
    encoder_surface.stem.2.0.weight: [128, 64, 3, 3], dtype=torch.float32
    encoder_surface.stem.2.0.bias: [128], dtype=torch.float32
    encoder_surface.stem.2.1.weight: [128], dtype=torch.float32
    encoder_surface.stem.2.1.bias: [128], dtype=torch.float32
    encoder_surface.stem.2.1.running_mean: [128], dtype=torch.float32
    encoder_surface.stem.2.1.running_var: [128], dtype=torch.float32
    encoder_surface.stem.2.1.num_batches_tracked: [], dtype=torch.int64
    encoder_surface.stem.3.log_p: [], dtype=torch.float32
    encoder_surface.fc.weight: [1024, 128], dtype=torch.float32
    encoder_surface.fc.bias: [1024], dtype=torch.float32
    encoder_curve.stem.0.0.weight: [32, 6, 3], dtype=torch.float32
    encoder_curve.stem.0.0.bias: [32], dtype=torch.float32
    encoder_curve.stem.0.1.weight: [32], dtype=torch.float32
    encoder_curve.stem.0.1.bias: [32], dtype=torch.float32
    encoder_curve.stem.0.1.running_mean: [32], dtype=torch.float32
    encoder_curve.stem.0.1.running_var: [32], dtype=torch.float32
  
  ================================================================================
  File: ts3d_graphclassification_5k_10epochs.ckpt (15.38 MB)
  ================================================================================
  
  📊 Checkpoint loaded successfully!
  Top-level keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
  
  ⚡ PyTorch Lightning Checkpoint
  Epoch: 9
  Global step: 140
  
  🔧 Hyperparameters:
    num_classes: 45
    metric_storage: MetricStorage
    return_embeddings: False
    graph_emb_dim: 128
  
  🏗️ Model Architecture:
  Total layers: 136
  Total parameters: 1,345,923
  
  📦 Module Groups (1 total):
  
    model (136 layers):
      model.curv_encoder.conv1.0.weight: [64, 6, 3] (torch.float32)
      model.curv_encoder.conv1.1.weight: [64] (torch.float32)
      model.curv_encoder.conv1.1.bias: [64] (torch.float32)
      model.curv_encoder.conv1.1.running_mean: [64] (torch.float32)
      model.curv_encoder.conv1.1.running_var: [64] (torch.float32)
  
  📝 Layer Types:
    Tensor: 136
  
  🔍 First 30 layers:
    model.curv_encoder.conv1.0.weight: [64, 6, 3], dtype=torch.float32
    model.curv_encoder.conv1.1.weight: [64], dtype=torch.float32
    model.curv_encoder.conv1.1.bias: [64], dtype=torch.float32
    model.curv_encoder.conv1.1.running_mean: [64], dtype=torch.float32
    model.curv_encoder.conv1.1.running_var: [64], dtype=torch.float32
    model.curv_encoder.conv1.1.num_batches_tracked: [], dtype=torch.int64
    model.curv_encoder.conv2.0.weight: [128, 64, 3], dtype=torch.float32
    model.curv_encoder.conv2.1.weight: [128], dtype=torch.float32
    model.curv_encoder.conv2.1.bias: [128], dtype=torch.float32
    model.curv_encoder.conv2.1.running_mean: [128], dtype=torch.float32
    model.curv_encoder.conv2.1.running_var: [128], dtype=torch.float32
    model.curv_encoder.conv2.1.num_batches_tracked: [], dtype=torch.int64
    model.curv_encoder.conv3.0.weight: [256, 128, 3], dtype=torch.float32
    model.curv_encoder.conv3.1.weight: [256], dtype=torch.float32
    model.curv_encoder.conv3.1.bias: [256], dtype=torch.float32
    model.curv_encoder.conv3.1.running_mean: [256], dtype=torch.float32
    model.curv_encoder.conv3.1.running_var: [256], dtype=torch.float32
    model.curv_encoder.conv3.1.num_batches_tracked: [], dtype=torch.int64
    model.curv_encoder.fc.linear.weight: [64, 256], dtype=torch.float32
    model.curv_encoder.fc.bn.weight: [64], dtype=torch.float32
    model.curv_encoder.fc.bn.bias: [64], dtype=torch.float32
    model.curv_encoder.fc.bn.running_mean: [64], dtype=torch.float32
    model.curv_encoder.fc.bn.running_var: [64], dtype=torch.float32
    model.curv_encoder.fc.bn.num_batches_tracked: [], dtype=torch.int64
    model.surf_encoder.conv1.0.weight: [64, 7, 3, 3], dtype=torch.float32
    model.surf_encoder.conv1.1.weight: [64], dtype=torch.float32
    model.surf_encoder.conv1.1.bias: [64], dtype=torch.float32
    model.surf_encoder.conv1.1.running_mean: [64], dtype=torch.float32
    model.surf_encoder.conv1.1.running_var: [64], dtype=torch.float32
    model.surf_encoder.conv1.1.num_batches_tracked: [], dtype=torch.int64
  ~~~



