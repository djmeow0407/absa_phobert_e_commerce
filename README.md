# PhoBERT Multi-Aspect Sentiment Analysis for Vietnamese Product Reviews

## Model Description

This model is a fine-tuned version of [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) for multi-aspect sentiment analysis on Vietnamese product reviews.

It can simultaneously predict sentiment across 8 different aspects from a single review text.

This model is used as the AI serving component for [E-Metric Hub](https://github.com/Ngnquoc1/E-Metric-Hub). The current version is intended for research/demo purposes and is not yet fully production-ready.

**Developed by:** [@djmeow0407](https://github.com/djmeow0407)  
**Model type:** Multi-label Text Classification  
**Language:** Vietnamese  
**Base model:** vinai/phobert-base  
**License:** MIT

## Supported Aspects

The model analyzes sentiment across these 8 aspects:
- **Price** (Giá cả)
- **Shipping** (Vận chuyển)
- **Outlook** (Ngoại hình/Hình thức)
- **Quality** (Chất lượng)
- **Size** (Kích cỡ)
- **Shop_Service** (Dịch vụ cửa hàng)
- **General** (Tổng quan)
- **Others** (Khác)

## Sentiment Labels

For each aspect, the model predicts one of four labels:
- **-1 (none)**: Aspect not mentioned in the review
- **0 (negative)**: Negative sentiment
- **1 (positive)**: Positive sentiment  
- **2 (neutral)**: Neutral sentiment

## Training Data

- **Training set:** 8,424 Vietnamese product reviews
- **Validation set:** 936 reviews
- **Test set:** 2,340 reviews
- **Domain:** E-commerce product reviews (primarily footwear)

The dataset exhibits significant class imbalance, with the `none` class being heavily dominant across all aspects.

## Training Procedure

### Training Hyperparameters

- **Optimizer:** AdamW
- **Learning rate:** 3e-5
- **Weight decay:** 0.01
- **Warmup ratio:** 0.1
- **Batch size:** 16 (training), 32 (evaluation)
- **Number of epochs:** 3
- **Max sequence length:** 256 tokens
- **Dropout:** 0.2
- **FP16 training:** Enabled (on GPU)
- **Random seed:** 123

### Model Architecture

The model consists of:
1. **PhoBERT backbone** for Vietnamese text encoding
2. **Dropout layer** (p=0.2) for regularization
3. **Linear classification head** that outputs logits for 8 aspects × 4 labels (32 outputs total)
4. **CrossEntropyLoss** with `ignore_index=-100` to handle missing aspect labels

## Performance

### Test Set Results (Best Model)

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.7014** |
| **Micro F1** | **0.9099** |
| **Overall Accuracy** | **90.99%** |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **none** | 0.9390 | 0.9702 | 0.9544 | 14,062 |
| **negative** | 0.6768 | 0.5587 | 0.6121 | 596 |
| **positive** | 0.8327 | 0.8714 | 0.8516 | 3,273 |
| **neutral** | 0.7518 | 0.2611 | 0.3876 | 789 |

### Key Observations

**Strengths:**
- Excellent performance on the dominant `none` class (F1: 0.95)
- Strong accuracy for `positive` sentiment (F1: 0.85)
- High overall accuracy (91%) and micro-F1 (0.91)

**Limitations:**
- Moderate performance on `negative` sentiment (F1: 0.61) due to class imbalance
- Lower recall on `neutral` class (26%), though precision is reasonable (75%)
- The neutral vs. none distinction remains challenging even for human annotators in some cases

## Usage

```python
# Luu mo hinh va thu inference tren review mau
FINAL_DIR = 'absa_phobert_model'
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
def predict_aspects(review_text: str):
    model.eval()
    encoded = tokenizer(review_text, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        preds = outputs.logits.argmax(dim=-1).squeeze(0).tolist()
    decoded = {aspect: INV_LABEL_MAP.get(int(pred), None) for aspect, pred in zip(ASPECT_COLUMNS, preds)}
    return decoded
sample_review = 'Giao hàng nhanh lắm ạ Giày đẹp chất lương rất ưng ý ạ hihijihih'
prediction = predict_aspects(sample_review)
print('Dự đoán cảm xúc trên từng khía cạnh:')
for aspect, value in prediction.items():
    print(f'{aspect}: {value}')
```

**Example Output:**

```
Dự đoán cảm xúc trên từng khía cạnh:
Price: -1
Shipping: 1
Outlook: 1
Quality: 1
Size: -1
Shop_Service: -1
General: -1
Others: -1
```

## Future Improvements

This model provides a strong foundation for multi-aspect sentiment analysis in Vietnamese. Two main directions for enhancement are planned:

### 1. Improving Model Accuracy (Especially Neutral & Negative Classes)

**Planned approaches:**

1. **Address class imbalance**
   - Implement oversampling or data augmentation for under-represented classes (`negative`, `neutral`)
   - Apply class-weighted loss or focal loss to penalize mistakes on minority classes more heavily

2. **Better handling of `none` vs `neutral` distinction**
   - Refine annotation guidelines to reduce ambiguity
   - Consider hierarchical classification: first detect if aspect is mentioned, then classify sentiment
   - Some cases are inherently ambiguous even for human annotators

3. **Model architecture improvements**
   - Experiment with multi-layer classification heads
   - Add attention mechanisms over aspects to share information
   - Explore multi-task learning setups

4. **Hyperparameter optimization**
   - Fine-tune learning rate, warmup schedule, and training epochs
   - Adjust dropout rates for better regularization
   - Implement early stopping based on macro-F1 score

### 2. Natural Language Explanation Layer

Beyond numeric predictions, the next major enhancement is to build an **NLP layer that generates natural-language explanations in Vietnamese**.

**Example:**

**Input review:**  
> "Giày đẹp, đi êm, ship nhanh, giá ổn."

**Model predictions:**
- Outlook = 1 (positive)
- Quality = 1 (positive)  
- Shipping = 1 (positive)
- Price = 1 (positive)
- Others = -1 (none)

**Generated explanation:**
> "Khách hàng hài lòng về hình thức, chất lượng, tốc độ giao hàng và giá cả; các khía cạnh khác không được đề cập."

This interpretability layer would make the model more practical for business applications like automated customer feedback analysis and review summarization.

## Acknowledgments

- Base model: [PhoBERT](https://github.com/VinAIResearch/PhoBERT) by VinAI Research

*Made with love by [@djmeow0407](https://github.com/djmeow0407)*  
