pytorch implementation of dynamic routing model by https://arxiv.org/pdf/1806.01501.pdf


### requirements
    - allennlp
    - pytorch

### command lines:
    ```bash
    allennlp train experiments/sst_5/sst_5_encoder_aggregator_classifier_lstm_dynamic_routing.json -s ./tmp/sst_5_0603 --include-package src
    ```

### authors
    - michael-wzhu
    - caolingyu