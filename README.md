### <p align="center"> BRPPNet: Balanced Privacy Protection Network for Referring Personal Image Privacy Protection
<br>
<div align="center">
  Jiacheng&nbsp;Lin</a> <b>&middot;</b>
  Xianwen&nbsp;Dai</a> <b>&middot;</b>
  Ke&nbsp;Nai</a> <b>&middot;</b>
  Jin&nbsp;Yuan</a> <b>&middot;</b>
  Xu&nbsp;Zhang</a> <b>&middot;</b>
  Zhiyong&nbsp;Li</a> <b>&middot;</b>
  Shutao&nbsp;Li</a>
  <br> <br>

  <a href="" target="_blank">Paper</a>
</div>

####

<br>
<p align="center">We will release code and proposed RPCOCO dataset in the future. </p>
<br>

<div align=center><img src="https://s2.loli.net/2023/05/03/WzA9goL8BQkU4Ys.png" /></div>

### Update
- 2023.05.03 Init repository.

### TODO List
- [ ] Code release. 

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

### BBCE Loss

The code of BBCE loss of Pytorch like as follows:

~~~
def Balanced_BCE_loss(scores, labels, eplison1=1.0, eplison2=-0.4, average=True):
    # Apply different weights to loss of positive samples and negative samples
    # Positive samples have the gradient weight of 1.0, while negative samples have the gradient weight of -0.4  
    
    # Classification loss as the average or the sum of balanced per-score loss
    cls_loss = F.binary_cross_entropy_with_logits(scores,labels)

    p = torch.sigmoid(scores) 
    pos_t = eplison1 * (1 - labels * p)
    neg_t = eplison2 * (1 - labels) * (1 - p)
    
    if average is True:
    
      bbce = torch.mean(cls_loss + (neg_t + pos_t))
      
    else:
    
      bbce = torch.sum(cls_loss + (neg_t + pos_t))
      
    return bbce
~~~

The code of BBCE Loss of Tensorflow like as follows:
~~~

def balanced_binary_cross_entropy(pred, mask, batch, epsilon1=1.0, epsilon2 = -0.4, average=False):
    
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
