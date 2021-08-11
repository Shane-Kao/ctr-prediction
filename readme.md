# Dataset
|Name|Row|Distinct|Column|Disk|Memory
--------------|:-----:|-----:|----:|----:|------------------------
|train.gz|40,428,967|40,428,967|24|1.04G|9.52G
|test.gz|4,577,464|4,577,464|23|118M|1.04G
|sampleSubmission.gz|4,577,464|4,577,464|2|33M|71M

- 無重複資料。
- 總共提供 24 個欄位，扣除 `id` 與 `click` (response) 後，有 22 個欄位可以用來產生特徵。

# Summary
|Fields|Description|Type|Missing|Distinct (train)|Distinct (test)
--------------|:-----:|-----:|----:|----:|------------------------
|id|Ad identifier|Continuous|0|40,428,967|4,577,464	
|click|0/1 for non-click/click (response)|Category|0|2	
|hour|"Format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC."|Continuous|0|240|24
|C1|Anonymized categorical variable|Category|0|7|7
|banner_pos| |Category|0|7|6	
|site_id| |Category|0|4,737|2,825	
|site_domain| |Category|0|7,745|3,366	
|site_category| |Category|0|26|22	
|app_id| |Category|0|8,552|3,952	
|app_domain| |Category|0|559|201	
|app_category| |Category|0|36|28	
|device_id| |Category|0|2,686,408|291,759	
|device_ip| |Category|0|6,729,486|1,077,199	
|device_model| |Category|0|8,251|5,438	
|device_type| |Category|0|5|4	
|device_conn_type| |Category|0|4|4	
|C14|	Anonymized categorical variableCategory|Category|0|2,626|1,257	
|C15|	Anonymized categorical variableCategory|Category|0|8|8	
|C16|	Anonymized categorical variableCategory|Category|0|9|9	
|C17|	Anonymized categorical variableCategory|Category|0|435|240	
|C18|	Anonymized categorical variableCategory|Category|0|4|4	
|C19|	Anonymized categorical variableCategory|Category|0|68|47	
|C20|	Anonymized categorical variableCategory|Category|0|172|162	
|C21|	Anonymized categorical variableCategory|Category|0|60|39

- 無缺失資料
- 大量的類別型資料，且 level 數高。

# Preprocessing
- `hour` 欄位抽取出最後兩位數(HH)，並視為 category。

# Feature Extraction
- Count Encoder
- Leave One Out
- Target Encoder
- Weight of Evidence
- CatBoost Encoder

# Feature Selection
- 對每一個 feature 計算 F-test 檢定量。
- 根據 F-test 結果，保留最顯著的 p% 個 feature。(p 視為參數，後續調參決定)

# Model Selection
- 訓練集樣本數: 2,000,000
- 線下測試集樣本數: 20,428,967
- CV: time series split
- CV: 3 fold
- CV: 驗證集樣本數: 200,000
- 模型: logistic regression

# Hyperparameter Tuning
- 參數1: 選擇 p% 個 feature
- 參數2: 模型有/無截距項
- 參數3: 模型使用 l1/l2 norm
- 隨機生成 5 個參數組合，並找出 CV score 最高的參數組合，當作最後訓練參數。

# Evaluation
|模型(encoder)|Test Score(線下)|Private Score|
--------------:|-----:|---
|loo|-0.40369|-0.40695
|caboost|-0.40381|-0.40722
|woe|-0.41770|
|target|-0.42464|
|count|-0.43191|
|loo + caboost| |**-0.40686**
- 根據線下測試集挑選出結果最好的兩個模型作融合。
- 融合結果: -0.40686 分數最高。
