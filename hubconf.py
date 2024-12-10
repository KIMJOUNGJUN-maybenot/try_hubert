# hubconf.py
dependencies = ["torch", "torchaudio", "transformers"]

from transformers import HubertForCTC

from transformers import HubertForCTC

def hubert_ee(model_path: str, vocab_size=50, ee_layers=None):
    """
    Load a huggingface HubertForCTC model from a local directory.
    """
    model = HubertForCTC.from_pretrained(model_path)
    # `vocab_size`와 `ee_layers`를 처리하는 로직 추가
    if ee_layers:
        model.config.layer_drop = ee_layers
    model.config.vocab_size = vocab_size
    model.eval()
    return model

