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
gradient propagation bias caused by unequal positive and negative pixels. To verify the effectiveness of BRPPNet, we manually construct a dataset “Refer-
ring Personal COCO” (RPCOCO). The experimental results demonstrate that BRPPNet outperforms the advanced approaches for RP-IPP, and the proposed
“Balanced-BCE loss” embedded into several existing approaches consistently
boosts performance, yielding remarkable improvements on all the metrics.
