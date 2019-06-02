### SST-5 CLASSIFIER

    - simple classifier
        1) cmd: 
        ```bash
        allennlp train experiments/sst_5/sst_5_simple_classifier_cnn_max_pool.json -s ./tmp/sst_5_0521 --include-package src
        ```
    
    - lstm + dynamic routing
        ```bash
        allennlp train experiments/sst_5/sst_5_encoder_aggregator_classifier_lstm_dynamic_routing.json -s ./tmp/sst_5_0531 --include-package src
        ```