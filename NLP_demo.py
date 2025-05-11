# %% [markdown]
# # MOAI 2025 實戰範例：基於預訓練 BERT 的文本分類

# %% [markdown]
# ## 1. 環境準備與庫引入
# 請確保你已經安裝了 PyTorch 和 Hugging Face 的 transformers 庫。
# 如果在 Kaggle 或 Colab 環境，這些通常已經安裝好。
# 如果是本地環境，請使用 pip 安裝：
# `pip install torch transformers pandas scikit-learn datasets`

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import random
import os

# 設定隨機數種子，確保結果可重現
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # 如果使用GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 檢查是否有可用的 GPU，並設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的設備: {device}")

# %% [markdown]
# ## 2. 數據準備
#
# 這裡我們模擬一個簡單的文本分類數據集。
# 在實際比賽中，數據可能會以 CSV 文件、JSON 文件或其他格式提供，你需要讀取並轉換為 Pandas DataFrame。
# 數據集需要包含文本列和標籤列。

# %%
# 模擬創建一個簡單的數據集
data = {
    'text': [
        "這個電影太棒了，我喜歡它！",
        "服務態度很差，食物也很難吃。",
        "這本書內容豐富，引人入勝。",
        "物流很慢，包裝破損。",
        "簡直完美！強烈推薦！",
        "體驗極差，不會再來。",
        "學到了很多新知識，很有啟發。",
        "質量一般，性價比不高。",
        "這真是一個愉快的經歷。",
        "完全浪費時間和金錢。"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 表示正面， 0 表示負面
}

df = pd.DataFrame(data)

print("原始數據集:")
print(df)

# 將數據集劃分為訓練集和驗證集
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label']) # stratify 保持標籤分佈一致

print(f"\n訓練集大小: {len(train_df)}")
print(f"驗證集大小: {len(val_df)}")

# %% [markdown]
# ## 3. 文本預處理：使用 BERT 的 Tokenizer
#
# BERT 模型需要特定的輸入格式。我們需要使用與預訓練模型配套的 tokenizer 來完成：
# 1. 將文本分割成 token (詞或詞片段)。
# 2. 將 token 轉換為模型能夠理解的數值 ID。
# 3. 添加特殊的 token (如 `[CLS]` 和 `[SEP]`)。
# 4. 對序列進行 padding (填充) 或 truncation (截斷)，使其達到固定的長度。
# 5. 生成 attention mask，告訴模型哪些是真實 token，哪些是 padding。

# %%
# 選擇一個預訓練模型名稱
# 'bert-base-chinese' 是一個常用的中文 BERT 模型
# 如果是英文文本，可以使用 'bert-base-uncased'
MODEL_NAME = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 設定最大序列長度。較長的文本需要更大的長度，但會增加計算量。
# 256 或 512 是常見的選擇。
MAX_LEN = 128 # 我們的模擬數據集文本很短，128 足夠了

def preprocess_text(text):
    """
    使用 tokenizer 處理單個文本
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,    # 添加 '[CLS]' 和 '[SEP]'
        max_length=MAX_LEN,         # 設定最大長度
        padding='max_length',       # 不足 MAX_LEN 的填充
        truncation=True,            # 超過 MAX_LEN 的截斷
        return_attention_mask=True, # 返回 attention mask
        return_tensors='pt',        # 返回 PyTorch 張量
    )
    # encode_plus 返回的是一個字典，包含 input_ids, attention_mask 等
    # 這裡我們只需要 input_ids, attention_mask (對於 BERT base 不需要 token_type_ids)
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0) # squeeze(0) 去掉 batch 維度

# 創建一個自定義的 PyTorch Dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 將張量從 (1, seq_len) 轉換為 (seq_len,)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        # 對於 BERT base 模型，通常不需要 token_type_ids，但有些模型或任務可能需要
        # token_type_ids = encoding['token_type_ids'].squeeze(0) if 'token_type_ids' in encoding else None


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'token_type_ids': token_type_ids, # 如果需要
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 創建訓練集和驗證集的 Dataset 實例
train_dataset = TextClassificationDataset(
    texts=train_df['text'].tolist(),
    labels=train_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = TextClassificationDataset(
    texts=val_df['text'].tolist(),
    labels=val_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# 設定 DataLoader 的 Batch Size
BATCH_SIZE = 8 # 對於 BERT 模型，Batch Size 通常比 CNN 小，因為模型更大更佔內存

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True # 訓練集需要打亂順序
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False # 驗證集不需要打亂
)

print("\n數據載入完成。")
# 查看一個 batch 的數據結構
batch = next(iter(train_loader))
print("\n一個 Batch 的數據結構:")
print(f"input_ids shape: {batch['input_ids'].shape}") # (batch_size, max_len)
print(f"attention_mask shape: {batch['attention_mask'].shape}") # (batch_size, max_len)
print(f"labels shape: {batch['labels'].shape}") # (batch_size,)


# %% [markdown]
# ## 4. 載入預訓練模型
#
# Hugging Face `transformers` 庫提供了 `AutoModelForSequenceClassification` 類，它可以方便地載入一個預訓練模型（例如 BERT），並在頂部添加一個用於分類的層。

# %%
# 載入預訓練模型，並指定類別數量 (num_labels)
# 我們的任務是二分類，所以 num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 將模型移動到指定的設備 (GPU 或 CPU)
model.to(device)

print("\n預訓練模型載入完成。")
print(f"模型結構:\n{model}")

# %% [markdown]
# ## 5. 設定訓練參數
#
# 定義優化器、學習率、訓練的 epoch 數量等。

# %%
# 定義優化器
optimizer = AdamW(model.parameters(), lr=5e-5) # AdamW 是訓練 Transformer 常用的優化器

# 定義訓練的 Epoch 數量
EPOCHS = 5 # 對於小型數據集和預訓練模型，幾個 Epoch 通常足夠

# 注意：對於分類任務，損失函數已經內嵌在 AutoModelForSequenceClassification 中了，
# 當你把 labels 作為 forward 的輸入時，模型會自動計算交叉熵損失並返回。
# 所以這裡不需要額外定義 `criterion = nn.CrossEntropyLoss()`

# %% [markdown]
# ## 6. 模型訓練
#
# 實現訓練循環。對於每個 epoch，遍歷訓練數據，進行前向傳播、計算損失、反向傳播和參數更新。

# %%
def train_epoch(model, data_loader, optimizer, device):
    """
    單個 epoch 的訓練函數
    """
    model.train() # 設定模型為訓練模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # 使用 tqdm 顯示進度條 (可選，需要安裝 tqdm: pip install tqdm)
    # from tqdm.auto import tqdm
    # data_loader = tqdm(data_loader, desc='Training')

    for batch in data_loader:
        # 將數據移動到設備
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向傳播
        # AutoModelForSequenceClassification 返回一個 Output 類型的對象
        # 其中包含了 loss 和 logits (原始預測分數)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels # 提供 labels 會自動計算損失
        )

        loss = outputs.loss
        logits = outputs.logits # 原始輸出分數

        # 計算準確率
        _, preds = torch.max(logits, dim=1) # 找到每個樣本最高分數對應的類別索引
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

        # 反向傳播
        loss.backward()

        # 更新參數
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples

    return avg_loss, accuracy.item()

# %% [markdown]
# ## 7. 模型評估
#
# 實現評估循環。在驗證集上評估模型性能，注意要使用 `model.eval()` 和 `torch.no_grad()`。

# %%
def evaluate(model, data_loader, device):
    """
    在驗證集上評估模型性能
    """
    model.eval() # 設定模型為評估模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # 不計算梯度
    with torch.no_grad():
        # from tqdm.auto import tqdm
        # data_loader = tqdm(data_loader, desc='Evaluating')

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向傳播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # 計算準確率
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples

    return avg_loss, accuracy.item()

# %% [markdown]
# ## 8. 執行訓練和評估
#
# 運行多個 Epoch 的訓練和評估。

# %%
best_val_accuracy = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n開始訓練...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 10)

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    print(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")

    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 可以根據驗證集準確率保存最好的模型
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        # torch.save(model.state_dict(), 'best_model_state_dict.pth') # 保存模型權重
        print("驗證集準確率提升，保存模型權重 (如果需要)...")


print("\n訓練結束。")
print(f"最佳驗證準確率: {best_val_accuracy:.4f}")

# %% [markdown]
# ## 9. 模型預測 (可選)
#
# 載入訓練好的模型，對新的文本進行預測。

# %%
def predict(text, model, tokenizer, max_len, device):
    """
    對單個文本進行預測
    """
    model.eval() # 設定模型為評估模式

    # 預處理文本
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt', # 返回 PyTorch 張量
    )

    # 將數據移動到設備
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # token_type_ids = encoding['token_type_ids'].to(device) if 'token_type_ids' in encoding else None # 如果需要

    with torch.no_grad():
        # 前向傳播
        # 如果沒有提供 labels，AutoModelForSequenceClassification 只返回 logits
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids # 如果需要
        )

        logits = outputs.logits # 獲取原始輸出分數

    # 獲取預測的類別概率 (Softmax)
    probabilities = torch.softmax(logits, dim=1)

    # 獲取預測的類別索引
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # 返回預測的類別索引和對應的概率
    return predicted_class_id, probabilities[0][predicted_class_id].item()


# 測試預測函數
test_text1 = "我非常喜歡這個產品，質量很好！"
test_text2 = "這次購物體驗很糟糕，非常失望。"

predicted_class1, probability1 = predict(test_text1, model, tokenizer, MAX_LEN, device)
print(f"\n文本: '{test_text1}'")
print(f"預測類別: {predicted_class1} (概率: {probability1:.4f})") # 0: 負面, 1: 正面

predicted_class2, probability2 = predict(test_text2, model, tokenizer, MAX_LEN, device)
print(f"\n文本: '{test_text2}'")
print(f"預測類別: {predicted_class2} (概率: {probability2:.4f})")

# %% [markdown]
# ## 總結與注意事項
#
# 這個範例展示了如何使用預訓練 BERT 模型進行文本分類。在 MOAI 比賽中，你可能需要根據具體任務進行修改：
#
# 1.  **數據集：** 替換數據讀取和處理部分，適應比賽提供的數據格式和內容。你可能需要處理缺失值、異常值等。
# 2.  **任務類型：** 如果不是二分類，修改 `num_labels` 參數，並確保數據集的標籤對應正確的類別索引。
# 3.  **模型選擇：** 如果是英文任務，使用 `bert-base-uncased` 或其他英文模型。對於中文任務，`bert-base-chinese` 是個不錯的起點。也可以嘗試其他更小的模型如 DistilBERT, RoBERTa 等，它們可能訓練更快。
# 4.  **超參數調整：** `MAX_LEN`, `BATCH_SIZE`, `EPOCHS`, `learning rate` 都可能需要根據數據集大小和計算資源進行調整。通常會從常見值開始嘗試。
# 5.  **評估指標：** 對於分類任務，除了準確率，Precision, Recall, F1-Score, ROC-AUC 等也是重要的評估指標，特別是當類別不平衡時。你可以使用 `sklearn.metrics` 來計算這些指標。
# 6.  **計算資源：** 訓練 BERT 模型需要 GPU。如果只有 CPU，訓練會非常慢。確保你的環境支持 GPU，並代碼能正確使用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
# 7.  **更複雜的任務：** 如果實戰題是物體識別或圖像分割，你需要應用 CNN 的知識（就像你準備圖像分類一樣），可能還會涉及到遷移學習（使用 ResNet, MobileNet 等預訓練模型）或特定的模型結構（YOLO, U-Net）。如果涉及到自然語言處理的生成任務或序列標註任務，模型結構和訓練方式會有所不同，但使用 `transformers` 庫載入預訓練模型的基本流程是相似的。
# 8.  **文件讀寫：** 比賽中可能要求你讀取特定格式的輸入文件，並將結果寫入特定格式的輸出文件。確保你熟悉 Python 的文件讀寫操作，以及如何使用 Pandas 處理 CSV 等結構化數據。

這個範例中的代碼註釋非常詳細，解釋了每一步的目的。你可以在理解這個範例的基礎上，針對比賽的具體要求進行修改和擴展。最重要的是理解**為什麼**要做這些步驟，而不是簡單地複製粘貼。

希望這個詳細的 NLP 實戰範例能幫助你更好地準備明天的比賽！祝你成功！
