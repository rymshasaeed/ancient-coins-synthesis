# Ancient Coins Reconstruction
The repository contains Conditional Generative Adversarial Network (CGAN) model, build using Pytorch, to classify and reconstruct the ancient coins. 

### Dataset
The dataset been used to train the model includes Hadrian Roman Imperial Coins obtained from publicly available <a href="http://numismatics.org/ocre/" target="blank_">Online Coins of the Roman Empire (OCRE)</a> database.
<p align="center">
  <img src="https://github.com/rimshasaeed/ancient-coins-synthesis/blob/main/results/dataset-sample.jpg", alt="dataset" width="50%">
  <br>
  <i>Dataset-Sample</i>
</p>

### Training Parameters
```
image size = [128, 128]
number of epochs = 2
generator dimensions = 100
batch size = 128
learning rate = 0.0002
```

### Results
The reconstructed samples could be obversed under <a href="https://github.com/rimshasaeed/ancient-coins-synthesis/tree/main/results/generated_images" target="blank_">this</a> directory. A better resolution synthesis could be achieved using the coin samples scaled greater than 128x128. 
