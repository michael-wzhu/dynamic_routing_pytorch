# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class SST5SimpleClassifierTest(ModelTestCase):
    def setUp(self):
        super(SST5SimpleClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/sst_5_simple_classifier.json',
                          'datasets/SST5/test.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
