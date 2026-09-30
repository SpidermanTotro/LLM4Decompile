"""Optional real-tensor regressions; require training dependencies."""
import importlib.util
import unittest

ML_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers', 'datasets'))


@unittest.skipUnless(ML_AVAILABLE, 'Install train/requirements.txt for real-tensor training checks')
class TrainingTokenTests(unittest.TestCase):
    def test_eos_is_attended_when_eos_is_padding_id(self):
        from train.finetune import DataCollatorForSupervisedDataset
        class Tokenizer:
            pad_token_id = 0
        collate = DataCollatorForSupervisedDataset(Tokenizer())
        batch = collate([{'input_ids': [2, 0], 'labels': [2, 0]},
                         {'input_ids': [3], 'labels': [3]}])
        self.assertEqual(batch['attention_mask'].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch['labels'].tolist(), [[2, 0], [3, -100]])

    def test_source_length_includes_real_eos(self):
        import torch
        from train.finetune import _tokenize_fn
        class Tokenizer:
            pad_token_id = 0
            model_max_length = 32
            def __call__(self, text, **kwargs):
                class Output:
                    input_ids = torch.tensor([[2, 0]])
                return Output()
        self.assertEqual(_tokenize_fn(['source'], Tokenizer())['input_ids_lens'], [2])
