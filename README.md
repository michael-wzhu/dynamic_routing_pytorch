# nas-classifier
searching for better text classifier using neural architecture search


### search algorithm, which is darts (possibly with a little modification)

0) macro network structure: 
    - branching: which is different from the cell search space from CV
    - 

1) genotypes: 

    - basic cnn, rnn
    - ops from transformer:
        - multi-head-attn: 先使用 allennlp 的 "multi_head_self_attention"
        - positionwise_feed_forward: "positionwise_feed_forward"
        
    - star-transformer
    
    - aggregators: dynamic-routing, self-attn, max-pool, avg-pool


### TODO
    - ADD L2 norm to selectively to a part of the model parameters
    - 
