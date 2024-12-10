from transformers import Wav2Vec2Processor, HubertForCTC

def download_model():
    model_name = "facebook/hubert-large-ls960-ft"
    save_path = "C:/Users/kimju/Desktop/hubert"

    # HuggingFace Hub에서 모델과 프로세서 다운로드
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    processor.save_pretrained(save_path)

    model = HubertForCTC.from_pretrained(model_name)
    model.save_pretrained(save_path)

    print(f"Model and processor saved to {save_path}")

if __name__ == "__main__":
    download_model()
