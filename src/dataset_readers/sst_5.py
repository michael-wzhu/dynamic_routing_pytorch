from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sst_5_data")
class SST5DatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the SST-5 data, and creates a
    dataset suitable for document (sentiment) classification using these data.

    Expected format for each input line: {"sentence": "text", "label": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the sentiment label.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentences into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                sample_json = json.loads(line)
                sentence = sample_json['sentence']
                label = sample_json['label']
                yield self.text_to_instance(sentence, label)

    @overrides
    def text_to_instance(self, sentence: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        fields = {'sentence': sentence_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
