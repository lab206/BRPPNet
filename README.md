### <p align="center"> BRPPNet: Balanced Privacy Protection Network for Referring Personal Image Privacy Protection
<br>
<div align="center">
  Jiacheng&nbsp;Lin</a> <b>&middot;</b>
  Xianwen&nbsp;Dai</a> <b>&middot;</b>
  Ke&nbsp;Nai</a> <b>&middot;</b>
  Jin&nbsp;Yuan</a> <b>&middot;</b>
  Zhiyong&nbsp;Li</a> <b>&middot;</b>  
  Xu&nbsp;Zhang</a> <b>&middot;</b>
  Shutao&nbsp;Li</a>
  <br> <br>

  <a href="https://www.sciencedirect.com/science/article/pii/S0957417423014628" target="_blank">Paper</a>
</div>

####


<div align=center><img src="assets/network.png" /></div>

### Update
- 2023.05.03 Init repository.
- 2023.06.05 Code and dataset release. 

### Abstract
Traditional personal image privacy protection usually suffers from the overprotection problem, where one or more undesired persons in an image may be
inevitably shielded, yielding unnecessary information loss. Motivated by this, this paper explores a novel task “Referring Personal Image Privacy Protection”
(RP-IPP) to protect the designated person in an image according to a user’s text or voice input. We propose a lightweight yet effective personal protection net-
work “Balanced Referring Personal PrivacyNet” (BRPPNet), which introduces a Multi-scale Feature Fusion Module (MFFM) with a proper “Balanced-BCE
loss” to effectively localize the referring person. Technically, MFFM adopts a lightweight CNN backbone to filter noise information as well as complement
visual features for high-quality mask generation. What is more, we have theoretically proven the insufficiency of binary cross-entropy (BCE) loss and its
variants for RP-IPP, which suffers from the serious imbalance problem during gradient propagation, and thus formulate “Balanced-BCE loss” to alleviate the
gradient propagation bias caused by unequal positive and negative pixels. To verify the effectiveness of BRPPNet, we manually construct a dataset “Referring Personal COCO” (RPCOCO). The experimental results demonstrate that BRPPNet outperforms the advanced approaches for RP-IPP, and the proposed
“Balanced-BCE loss” embedded into several existing approaches consistently
boosts performance, yielding remarkable improvements on all the metrics.


## Installation

1. Environment:

- Python 3.7
- tensorflow 1.15

2. Dataset preparation
- Put the folder of COCO training set ("`train2014`") under `data/images/`.
- Running the script for data preparation under `data/`:
 ```
cd data
python data_process_v2.py --data_root . --output_dir data_v2 --dataset [rpcocos/rpcocom/rpcocol] --split [unc/umd/google] --generate_mask
 ```
## Training and Evaluating

1. Pretrained Backbones
   We use the backbone weights proviede by [VLT](https://github.com/henghuiding/Vision-Language-Transformer).
2. Specify hyperparameters, dataset path and pretrained weight path in the configuration file. Please refer to the examples under `/config`, or config file of our pretrained models.
3. Training 
```
python main.py train [PATH_TO_CONFIG_FILE]
```
 4. Evaluating
```
python main.py test [PATH_TO_CONFIG_FILE]
```

### Balanced Binary Cross Entropy Loss

The code of BBCE Loss of Tensorflow like as follows:
~~~
def Balanced_binary_cross_entropy(pred, mask, batch, epsilon1=1.0, epsilon2 = -0.4, average=False):
    
    # Apply different weights to loss of positive samples and negative samples
    # Positive samples have the gradient weight of 1.0, while negative samples have the gradient weight of -0.4
    
    # Classification loss as the average or the sum of balanced per-score loss
    sig_pred = tf.nn.sigmoid(pred)
    pos_t = epsilon1 * (1 - mask * sig_pred)
    neg_t = epsilon2 * (1 - mask) * (1 - sig_pred)
    BCE = K.binary_crossentropy(mask, pred, from_logits=True)
    if average is True:
    
      BBCE = K.mean((BCE + (neg_t+pos_t)))
      
    else:
    
      BBCE = K.sum((BCE + (neg_t+pos_t)))
    
    bbce_loss = BBCE / batch
    
    return bbce_loss
~~~

### Citation
~~~bibtex
@article{LIN2023120960,
  title={{BRPPNet:} {Balanced} privacy protection network for referring personal image privacy protection},
  author={Lin, Jiacheng and others},
  journal={Expert Syst. Appl.},
  pages={120960},
  year={2023},
  publisher={Elsevier}
}
~~~


### Acknowledgement
Our project is developed based on [MCN](https://github.com/luogen1996/MCN), [VLT](https://github.com/henghuiding/Vision-Language-Transformer). Thanks for their excellence works.
