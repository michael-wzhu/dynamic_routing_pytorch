# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from src.dataset_readers.sst_5 import SST5DatasetReader


class TestSST5DatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SST5DatasetReader()
        instances = ensure_list(reader.read('datasets/SST5/test.jsonl'))

        instance3 = {
            "sentence": ["Cute", ",", "funny", ",", "heartwarming", "digitally", "animated", "feature", "film", "with",
                         "plenty",
                         "of", "slapstick", "humor", "for", "the", "kids", ",", "lots", "of",
                         "in", "-", "jokes", "for", "the", "adults", "and", "heart",
                         "enough", "for", "everyone", ".", ],
            "label": "very pos"
        }



        # instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
        #              "abstract": ["We", "investigate", "various", "contextual", "effects"],
        #              "venue": "ACL"}
        #
        # instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
        #              "abstract": ["Execut", "ion", "of", "classical", "plans"],
        #              "venue": "AI"}
        #
        # instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
        #                        "Traveller", "Problem"],
        #              "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
        #              "venue": "AI"}

        assert len(instances) == 2209
        print(instances[0].fields["sentence"].tokens)
        print(instances[1].fields["sentence"].tokens)
        print(instances[2].fields["sentence"].tokens)

        fields = instances[2].fields
        assert [t.text for t in fields["sentence"].tokens] == instance3["sentence"]
        assert fields["label"].label == instance3["label"]
