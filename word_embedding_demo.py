# %% [markdown]
# # 範例 2: 使用預訓練詞向量初始化 Embedding 層
#
# 這個筆記本展示了：
# 1. 模擬加載預訓練詞向量（如 Word2Vec 或 GloVe 格式）。
# 2. 創建詞彙表 (Vocabulary) 和詞到索引的映射。
# 3. 構建一個 Embedding 矩陣。
# 4. 使用 Embedding 矩陣初始化 PyTorch 的 `nn.Embedding` 層。
# 5. 構建一個簡單的文本分類模型，使用詞向量的平均值作為文本表示。
# 6. 簡單的訓練和預測示例。

# %% [markdown]
# ## 1. 引入所需的庫

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import os
import pandas as pd # 模擬數據用

# 設定隨機數種子
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的設備: {device}")


# %% [markdown]
# ## 2. 模擬數據集和預訓練詞向量
#
# 我們模擬一個包含少量文本和對應類別標籤的數據集，以及一個小型的「預訓練」詞向量字典。
#
# 在實際比賽中，你可能需要讀取 CSV/JSON 文件來獲取文本和標籤，並讀取 Word2Vec (`.vec` 或 `.bin`) 或 GloVe (`.txt`) 文件來獲取詞向量。

# %%
# 模擬數據集
data = {
    'text': [
        "great movie enjoyed it",
        "bad service food was awful",
        "interesting book lots of info",
        "slow delivery package damaged",
        "perfect highly recommend",
        "terrible experience never again",
        "learned many new things",
        "quality so so not worth price",
        "a truly pleasant experience",
        "total waste of time and money"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1:正面, 0:負面
}
df = pd.DataFrame(data)

# 簡單分詞 (假設已預處理，這裡直接按空格分)
df['tokens'] = df['text'].apply(lambda x: x.split())

print("模擬數據集 (已分詞):")
print(df)

# 模擬預訓練詞向量
# 格式：詞語 -> 向量 (numpy 數組或列表)
# 在實際中，這會從文件中讀取
simulated_word_vectors = {
    "great": np.array([0.1, 0.2, 0.3, 0.4]),
    "movie": np.array([0.5, 0.6, 0.7, 0.8]),
    "enjoyed": np.array([0.9, 1.0, 1.1, 1.2]),
    "service": np.array([-0.1, -0.2, -0.3, -0.4]),
    "awful": np.array([-0.5, -0.6, -0.7, -0.8]),
    "interesting": np.array([0.2, 0.3, 0.4, 0.5]),
    "book": np.array([0.6, 0.7, 0.8, 0.9]),
    "info": np.array([1.0, 1.1, 1.2, 1.3]),
    "slow": np.array([-0.2, -0.3, -0.4, -0.5]),
    "damaged": np.array([-0.6, -0.7, -0.8, -0.9]),
    "perfect": np.array([0.3, 0.4, 0.5, 0.6]),
    "recommend": np.array([0.7, 0.8, 0.9, 1.0]),
    "terrible": np.array([-0.3, -0.4, -0.5, -0.6]),
    "experience": np.array([-0.7, -0.8, -0.9, -1.0]),
    "never": np.array([-1.1, -1.2, -1.3, -1.4]),
    "learned": np.array([0.4, 0.5, 0.6, 0.7]),
    "things": np.array([0.8, 0.9, 1.0, 1.1]),
    "quality": np.array([-0.4, -0.5, -0.6, -0.7]),
    "worth": np.array([-0.8, -0.9, -1.0, -1.1]),
    "price": np.array([-1.2, -1.3, -1.4, -1.5]),
    "truly": np.array([0.5, 0.6, 0.7, 0.8]),
    "pleasant": np.array([0.9, 1.0, 1.1, 1.2]),
    "total": np.array([-0.5, -0.6, -0.7, -0.8]),
    "waste": np.array([-0.9, -1.0, -1.1, -1.2]),
    "time": np.array([-1.3, -1.4, -1.5, -1.6]),
    "money": np.array([-1.7, -1.8, -1.9, -2.0]),
    # 模擬一些數據集裡有但詞向量裡沒有的詞 (Out-of-Dictionary, OOD)
    "it": np.array([0.1, 0.1, 0.1, 0.1]),
    "this": np.array([0.2, 0.2, 0.2, 0.2]),
    "an": np.array([0.3, 0.3, 0.3, 0.3]),
    "is": np.array([0.4, 0.4, 0.4, 0.4]),
    "a": np.array([0.5, 0.5, 0.5, 0.5]),
    "and": np.array([0.6, 0.6, 0.6, 0.6]),
    "of": np.array([0.7, 0.7, 0.7, 0.7]),
    "was": np.array([0.8, 0.8, 0.8, 0.8]),
    "not": np.array([0.9, 0.9, 0.9, 0.9]),
    "so": np.array([1.0, 1.0, 1.0, 1.0]),
    "lots": np.array([1.1, 1.1, 1.1, 1.1]),
    "info": np.array([1.2, 1.2, 1.2, 1.2]),
    "package": np.array([1.3, 1.3, 1.3, 1.3]),
    "many": np.array([1.4, 1.4, 1.4, 1.4]),
    "new": np.array([1.5, 1.5, 1.5, 1.5]),
}
EMBEDDING_DIM = list(simulated_word_vectors.values())[0].shape[0] # 獲取詞向量維度

print(f"\n模擬詞向量維度: {EMBEDDING_DIM}")


# %% [markdown]
# ## 3. 構建詞彙表 (Vocabulary)
#
# 創建一個從詞語到唯一整數索引的映射。我們需要包含數據集中的所有詞語，以及處理未知詞 `<unk>` 和 Padding 詞 `<pad>` 的特殊標記。

# %%
# 收集數據集中的所有唯一詞語
all_words = set()
for tokens in df['tokens']:
    all_words.update(tokens)

# 創建詞語到索引的映射 (word2idx)
# 首先添加特殊標記
word2idx = {"<pad>": 0, "<unk>": 1}
# 然後添加數據集中的詞語
for word in sorted(list(all_words)): # 按字母順序排序以便穩定性
    if word not in word2idx:
        word2idx[word] = len(word2idx)

# 創建索引到詞語的映射 (idx2word)
idx2word = {idx: word for word, idx in word2idx.items()}

VOCAB_SIZE = len(word2idx)
print(f"\n詞彙表大小 (包含 <pad> 和 <unk>): {VOCAB_SIZE}")
print(f"詞彙表示例: {list(word2idx.items())[:10]}...")

# %% [markdown]
# ## 4. 創建 Embedding 矩陣
#
# 根據詞彙表和預訓練詞向量，構建一個形狀為 `(vocab_size, embedding_dim)` 的矩陣。
# - 矩陣的每一行對應詞彙表中的一個詞。
# - 對於詞彙表中在預訓練詞向量字典中存在的詞，使用其預訓練向量。
# - 對於 `<pad>` 詞，通常設置為零向量。
# - 對於 `<unk>` 詞或詞彙表中但在預訓練詞向量字典中不存在的詞 (OOD 詞)，可以設置為零向量，或者更常見地，隨機初始化一個向量。

# %%
# 創建一個初始化為零的 Embedding 矩陣
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

# 為 <unk> 詞隨機初始化一個向量 (也可以是零向量)
# 使用與 PyTorch 默認初始化Embedding 層相似的範圍
unk_vector = np.random.uniform(-np.sqrt(3/EMBEDDING_DIM), np.sqrt(3/EMBEDDING_DIM), EMBEDDING_DIM)
embedding_matrix[word2idx["<unk>"]] = unk_vector

# 填充 Embedding 矩陣
oov_count = 0 # 計數有多少詞語在詞彙表中但不在預訓練詞向量中 (Out Of Vocabulary)
for word, idx in word2idx.items():
    if word == "<pad>" or word == "<unk>":
        continue # 特殊標記已處理

    if word in simulated_word_vectors:
        embedding_matrix[idx] = simulated_word_vectors[word]
    else:
        # 如果詞語在詞彙表中但不在模擬的預訓練向量中
        embedding_matrix[idx] = unk_vector # 使用 <unk> 向量
        oov_count += 1
        # print(f"Warning: '{word}' not in simulated pre-trained vectors, using <unk> vector.") # 可選打印

print(f"\n在詞彙表中但在模擬預訓練向量中找不到的詞數量 (OOD): {oov_count}")
print(f"Embedding 矩陣形狀: {embedding_matrix.shape}")

# 將 numpy 矩陣轉換為 torch.Tensor
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

# %% [markdown]
# ## 5. 將文本序列轉換為索引序列並 Pad
#
# 模型需要數字輸入，所以需要將分詞後的文本序列轉換為對應的索引序列。同時，為了解決 Batch 中序列長度不一致的問題，我們需要對序列進行 Padding，使其達到一個固定的最大長度。

# %%
MAX_LEN = 10 # 設定一個最大序列長度 (比最長文本的詞數長一些即可)

def tokens_to_indices(tokens, word2idx, max_len, pad_idx, unk_idx):
    """
    將詞語列表轉換為索引列表，進行 Padding 和 Truncation。
    """
    indices = [word2idx.get(token, unk_idx) for token in tokens] # 查找索引，找不到使用 unk_idx

    # Truncation (截斷)：如果序列長度超過 max_len，截斷
    if len(indices) > max_len:
        indices = indices[:max_len]

    # Padding：如果序列長度不足 max_len，用 pad_idx 填充
    if len(indices) < max_len:
        indices.extend([pad_idx] * (max_len - len(indices)))

    return indices

# 對 DataFrame 中的所有 tokens 序列進行轉換和 Padding
df['indices'] = df['tokens'].apply(
    lambda tokens: tokens_to_indices(tokens, word2idx, MAX_LEN, word2idx["<pad>"], word2idx["<unk>"])
)

print("\n轉換為索引序列並 Padding 後:")
print(df)

# 查看一個索引序列示例
print(f"\n第一個文本的索引序列: {df['indices'].iloc[0]}")

# %% [markdown]
# ## 6. 創建 PyTorch Dataset 和 DataLoader
#
# 將處理後的索引序列和標籤打包成 Dataset，並創建 DataLoader，以便批量訓練。

# %%
class TextClassificationDataset(Dataset):
    def __init__(self, indices_list, labels):
        # 將索引列表和標籤轉換為 PyTorch 張量
        self.indices = torch.tensor(indices_list, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]

# 創建 Dataset 實例
# 我們這裡沒有劃分訓練集和驗證集，直接使用全部數據進行演示
dataset = TextClassificationDataset(df['indices'].tolist(), df['label'].tolist())

# 設定 DataLoader 的 Batch Size
BATCH_SIZE = 4

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\n數據載入完成。")
print(f"數據集大小: {len(dataset)}")
print(f"每個 Batch 大小: {BATCH_SIZE}")

# 查看一個 batch 的數據結構
batch_indices, batch_labels = next(iter(dataloader))
print("\n一個 Batch 的數據結構:")
print(f"索引序列 Batch shape: {batch_indices.shape}") # (batch_size, max_len)
print(f"標籤 Batch shape: {batch_labels.shape}")       # (batch_size,)


# %% [markdown]
# ## 7. 定義一個簡單的模型
#
# 構建一個包含 Embedding 層、平均池化和 Linear 層的簡單模型。

# %%
class SimpleEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super().__init__()

        # 使用預訓練的 Embedding 矩陣初始化 Embedding 層
        # freeze=False 允許在訓練過程中微調詞向量
        # freeze=True 則固定詞向量不更新
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Embedding 維度等於傳入的 embedding_matrix 的第二個維度
        embedding_dim = embedding_matrix.size(1)

        # 定義一個線性分類層
        # 輸入維度是每個文本的平均詞向量維度 (embedding_dim)
        # 輸出維度是類別數
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x 是輸入的索引序列張量 (Batch, SeqLen)
        embedded = self.embedding(x) # 輸出 (Batch, SeqLen, EmbeddingDim)

        # 對序列維度 (SeqLen) 進行平均池化
        # 簡單地計算每個文本中所有詞向量的平均值作為文本表示
        # 注意：對於 padding 位置的零向量，它們不會影響平均值 (假設填充值為 0，對應的向量也是 0)
        # 如果 pad_idx 對應的向量不是零向量，或者想忽略 padding 位置，需要更複雜的處理
        pooled = torch.mean(embedded, dim=1) # 輸出 (Batch, EmbeddingDim)

        # 將平均後的文本表示傳入線性分類層
        logits = self.fc(pooled) # 輸出 (Batch, NumClasses)

        return logits

# 實例化模型
NUM_CLASSES = 2
model = SimpleEmbeddingClassifier(embedding_matrix, NUM_CLASSES)
model.to(device) # 將模型移動到設備

print("\n簡單模型定義完成:")
print(model)

# %% [markdown]
# ## 8. 設定損失函數和優化器

# %%
criterion = nn.CrossEntropyLoss() # 用於分類任務的交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用 Adam 優化器

# %% [markdown]
# ## 9. 訓練循環 (簡單示例)
#
# 在數據集上進行幾個 Epoch 的訓練。這只是一個非常簡化的訓練流程。

# %%
EPOCHS = 10 # 訓練 Epoch 數量

print("\n開始訓練 (簡單示例)...")

for epoch in range(EPOCHS):
    model.train() # 設定模型為訓練模式
    running_loss = 0.0

    for inputs, labels in dataloader:
        # 將數據移動到設備
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs) # 輸出 logits

        # 計算損失
        loss = criterion(outputs, labels)

        # 反向傳播
        loss.backward()

        # 更新參數
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) # 累積每個 batch 的總損失

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

print("\n訓練結束。")

# %% [markdown]
# ## 10. 模型預測 (示例)
#
# 使用訓練好的模型對新的文本進行預測。

# %%
def predict_text(text, model, word2idx, max_len, pad_idx, unk_idx, device):
    """
    對單個原始文本進行預測。
    """
    model.eval() # 設定模型為評估模式

    # 預處理文本：小寫，移除標點，分詞 (這裡使用簡單分詞)
    text_lower = text.lower()
    text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
    tokens = text_no_punct.strip().split()

    # 轉換為索引序列並 Padding
    indices = tokens_to_indices(tokens, word2idx, max_len, pad_idx, unk_idx)

    # 轉換為 PyTorch 張量，並添加 Batch 維度
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad(): # 預測時不計算梯度
        outputs = model(input_tensor) # 輸出 logits (1, NumClasses)

    # 獲取預測概率 (使用 Softmax)
    probabilities = torch.softmax(outputs, dim=1)

    # 獲取預測類別索引
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # 返回預測的類別索引和概率
    return predicted_class_id, probabilities[0]


# 測試預測函數
test_text1 = "This is a very good movie!"
test_text2 = "I hate the slow service."

predicted_class1, probs1 = predict_text(test_text1, model, word2idx, MAX_LEN, word2idx["<pad>"], word2idx["<unk>"], device)
print(f"\n測試文本 1: '{test_text1}'")
print(f"預測類別: {predicted_class1} (概率: {probs1.tolist()})") # 0:負面, 1:正面 (取決於模型訓練結果)

predicted_class2, probs2 = predict_text(test_text2, model, word2idx, MAX_LEN, word2idx["<pad>"], word2idx["<unk>"], device)
print(f"\n測試文本 2: '{test_text2}'")
print(f"預測類別: {predicted_class2} (概率: {probs2.tolist()})")

# %% [markdown]
# ## 11. 總結與注意事項
#
# - **核心思想:** 將詞語映射到預訓練向量，然後使用這些向量作為模型的輸入特徵。`nn.Embedding.from_pretrained` 是關鍵。
# - **處理 OOD 詞:** 對於在預訓練詞向量中不存在的詞，如何處理很重要。使用 `<unk>` 標記並隨機初始化其向量是一個常見做法。
# - **Padding:** 對變長序列進行 Padding 是 Batch 處理的必要步驟。確保模型能正確處理 Padding（例如，Padding 位置對應的向量通常是零，並且在池化等操作中需要注意不要讓它們影響結果，儘管簡單平均在 Padding 向量為零時影響較小）。
# - **文本表示:** 這個範例使用了簡單的平均詞向量作為文本表示。更複雜的模型會使用 RNNs (如 LSTM, GRU) 或 Transformer 來更好地捕捉詞語之間的順序和相互作用。
# - **實際文件加載:** 在實際比賽中，你需要使用 `gensim` 等庫來加載 Word2Vec 或 GloVe 文件。例如 `from gensim.models import KeyedVectors; word_vectors = KeyedVectors.load_word2vec_format('path/to/your/vectors.vec', binary=False)`。
# - **模型複雜度:** 這個模型非常簡單，僅用於演示目的。實際的文本分類任務通常需要更深或更複雜的網絡結構。
# - **Fine-tuning:** `freeze=False` 允許詞向量在訓練過程中與模型其他參數一起更新，這稱為 fine-tuning。如果數據集較小或詞向量質量很高，可以考慮設置 `freeze=True`。
#
# 這兩個筆記本範例應該能讓你對文本預處理和利用預訓練詞向量初始化 Embedding 層有一個清晰的了解。在 MOAI 比賽中，如果遇到相關問題，可以根據這些基礎進行適應和擴展。
