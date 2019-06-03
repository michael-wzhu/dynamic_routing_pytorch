### SST-5 CLASSIFIER

    - lstm + dynamic routing
        ```bash
        allennlp train experiments/sst_5/sst_5_encoder_aggregator_classifier_lstm_dynamic_routing.json -s ./tmp/sst_5_0531 --include-package src
        ```