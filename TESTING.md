# Tesing wav2vec2 and wav2vec2-bert inference

## Models for testing

```code
cd ~
git clone https://huggingface.co/Korla/Wav2Vec2BertForCTC-hsb
git clone  https://huggingface.co/Korla/omniASR_W2V_300M_hsb
```

## Model conversion to GGML

### Preparation

```code
cd wav2vec2.cpp/
rm -rf pythonenv
python3 -m venv pythonenv/
cd pythonenv
source bin/activate
pip3 install torch transformers
```

### Model conversion

```code
python ../models/convert-wav2vec2-to-ggml.py \
    ~/omniASR_W2V_300M_hsb/ \
    ../models/Korla-omniASR_W2V_300M_hsb

python ../models/convert-wav2vec2bert-to-ggml.py \
    ~/Wav2Vec2BertForCTC-hsb/ \
    ../models/Korla-Wav2Vec2BertForCTC-hsb

    
    
```
