# %% [markdown]
# # 範例 1: 文本基本預處理
#
# 這個筆記本展示了如何使用 Python 進行常見的文本預處理步驟，包括：
# 1. 小寫轉換
# 2. 移除標點符號
# 3. 簡單的分詞

# %% [markdown]
# ## 1. 引入所需的庫
# 我們主要使用 Python 的內建功能和 `re` (正則表達式) 模塊。

# %%
import re
import pandas as pd # 常用於數據處理，雖然這裡只有少量文本，用 pandas DataFrame 方便展示

# %% [markdown]
# ## 2. 模擬原始文本數據
# 假設我們有一些包含標點符號、大小寫混雜的原始文本。

# %%
raw_texts = [
    "Hello, World! This is an Example.",
    "NLP is interesting. Is it useful?",
    "Let's try: Punctuation removal test!",
    "  Whitespace issues?  ",
    "Another line with numbers 123 and symbols #@$%."
]

# 將其放入 DataFrame 更像實際比賽場景
df = pd.DataFrame({'text': raw_texts})

print("原始 DataFrame:")
print(df)

# %% [markdown]
# ## 3. 小寫轉換
# 將所有文本轉換為小寫是常見的第一步，這樣 "Hello" 和 "hello" 會被視為同一個詞。

# %%
# 使用 apply 和 lambda 函數對 DataFrame 的 'text' 列進行操作
df['text_lower'] = df['text'].apply(lambda x: x.lower())

print("\n小寫轉換後:")
print(df)

# %% [markdown]
# ## 4. 移除標點符號
# 我們使用正則表達式來匹配並替換標點符號。常見的做法是只保留字母、數字和空格。

# %%
# 定義一個正則表達式，匹配所有不是字母、數字或空格的字符
# ^\w\s 表示非單詞字符 (\w 是字母、數字、下劃線) 或非空白字符 (\s 是空格、tab、換行)
# 更簡單且常用的正則表達式是 [^\w\s]
# [^\w\s] 會匹配所有非字母、非數字、非下劃線、非空白字符
# 如果你只希望保留字母和數字，可以使用 [^a-zA-Z0-9\s]
punctuation_pattern = re.compile(r'[^\w\s]')

def remove_punctuation(text):
    # 使用 sub 方法將匹配到的標點符號替換為空字符串
    return punctuation_pattern.sub('', text)

df['text_no_punct'] = df['text_lower'].apply(remove_punctuation) # 對小寫後的文本進行處理

print("\n移除標點符號後:")
print(df)


# 注意：上面的方法也會移除下劃線。如果你想保留下劃線，需要調整正則表達式。
# 例如，只移除常見標點符號可以使用 string.punctuation
import string
def remove_common_punctuation(text):
     return text.translate(str.maketrans('', '', string.punctuation))

df['text_no_common_punct'] = df['text_lower'].apply(remove_common_punctuation)
print("\n移除常見標點符號後 (使用 string.punctuation):")
print(df)
# 在比賽中，根據具體需求選擇合適的方法和正則表達式。我們繼續使用 remove_punctuation 的結果。

# %% [markdown]
# ## 5. 簡單分詞 (Tokenization)
# 將清洗後的文本分割成單詞或詞塊列表。最簡單的方法是按空格分割。

# %%
def simple_tokenize(text):
    # 按空格分割字符串
    # strip() 移除首尾空白，避免 split() 產生空字符串
    return text.strip().split()

df['tokens'] = df['text_no_punct'].apply(simple_tokenize)

print("\n簡單分詞後:")
print(df)

# %% [markdown]
# ## 6. 總結與注意事項
#
# - **基本步驟:** 小寫轉換、移除標點、分詞是文本預處理的常見開端。
# - **更高級的預處理:** 在實際 NLP 任務中，可能還需要：
#     - **移除停用詞 (Stop Words):** 移除 "a", "the", "is" 等常見且對任務幫助不大的詞。需要一個停用詞列表。
#     - **詞形還原 (Stemming/Lemmatization):** 將詞語還原到詞根或原形 (e.g., "running", "ran" -> "run")。需要 NLTK 或 SpaCy 等庫。
#     - **處理數字和特殊字符:** 如何處理數字 (保留, 替換為 <NUM>)，如何處理特定的表情符號、網址等。
#     - **處理空白符:** 如示例中的多餘空格。簡單的 `strip()` + `split()` 通常能處理。
# - **分詞器的選擇:** 對於中文或其他語言，簡單按空格分詞可能不夠。需要使用專門的分詞工具（如 Jieba 對於中文）。對於基於 Transformer 的預訓練模型，通常直接使用其配套的 `Tokenizer` (如 Hugging Face `transformers` 提供的) 進行處理，這一步已經包含了將文本轉換為模型所需的 ID 序列。
#
# 在 MOAI 比賽中，如果要求你使用預訓練模型，很可能直接讓你使用該模型配套的 Tokenizer。如果題目是從零開始或者使用傳統方法，則需要你自己實現或使用庫進行這些預處理步驟。理解這些基本概念是重要的。
