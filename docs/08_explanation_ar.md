# شرح المهام المنجزة — استجابة لمتطلبات الأستاذ المشرف

## ما الذي يريده الدكتور؟

يريد الأستاذ المشرف ترقية الورقة العلمية إلى مستوى مجلة **Q1**. وتتضمن الخطة البحثية تنفيذ **10 مهام** مقسمة على **3 أولويات**، بالإضافة إلى سؤالين توضيحيين.

### الأولوية 1 (عاجلة — بدون تدريب جديد، تُرسل أولاً)
1. **الحقيقة الأرضية البديلة (Alternative Ground Truth)**: إعادة وسم النصوص المعاد صياغتها باسم نموذج إعادة الصياغة نفسه بدل المؤلف الأصلي، وإعادة حساب الدقة و Macro-F1 لكل نماذج إعادة الصياغة الستة — لإثبات تأثير **Fingerprint-Overwrite**.
2. **مصدر مجموعة الأخبار الثانية (CNN/DailyMail)**: توثيق مصدر النصوص البشرية وطريقة توليد النسخ الآلية (النماذج، المزودات، البرومبتات، إعدادات التوليد)، مع حصر نتائج التقييم (Zero-Shot).
3. **مصفوفة التباس RoBERTa** تحت هجوم إعادة الصياغة بـ Gemma.

### الأولوية 2 (جهد متوسط — مع إعادة تدريب)
4. **مقدار التشويه (Perturbation Magnitude)**: حساب BERTScore و Self-BLEU بين الأصل وإعادة الصياغة لكل نموذج.
5. **توزيع الأصناف المتوقعة (Prediction Distribution)** للنموذج الهجين تحت الأخطاء الإملائية (5% و20%) والترجمة.
6. **إعادة تدريب النموذج الهجين بدون Gemma** (استئصال 6 أصناف) ثم إعادة التقييم.
7. **بروتوكول RoBERTa بخمس بذور** (42، 123، 999، 2024، 8080) للثنائي والمتعدد.

### الأولوية 3 (تحليلية)
8. **مسافات المراكز في فضاء الخصائص (Centroid Distances)**.
9. **زمن الاستدلال (Inference Latency Breakdown)**.
10. **تحليل الثقة (Confidence Calibration Analysis)**.

### السؤالان التوضيحيان
- **مجموعة WP**: هي Reddit [r/WritingPrompts](https://www.reddit.com/r/WritingPrompts/) وليست ويكيبيديا.
- **عدد العينات (Essay 396 و WP 398 بدل 400)**: نتج عن استبعاد الأسطر ذات القيم الفارغة `dropna()` أثناء هندسة الخصائص في [`notebooks/data_engineer.ipynb`](../notebooks/data_engineer.ipynb).

---

## ما الذي تم إنجازه؟ (النتائج والملفات بالتفصيل)

### ✅ المهمة 1 — الحقيقة الأرضية البديلة (Fingerprint-Overwrite)
* **المطلوب**: إعادة الوسم باسم المُعاد صياغته وإعادة حساب مقاييس الأداء.
* **ما تم تنفيذه**: تشغيل نموذج XGBoost المحفوظ `Multi_BERT` على ملفات إعادة الصياغة الستة ومقارنة الحقيقة الأصلية مقابل البديلة.
* **النتيجة**: تأثير Fingerprint-Overwrite **مؤكد لنموذجي Gemma فقط** (الدقة البديلة 81.5% و77.0% مقابل 25–27% بالأصلية). أما Qwen وMistral وLLaMA وGPT-4o-mini فتنهار التنبؤات فيها نحو Gemma-2-9B (3–7% فقط نحو الصنف الحقيقي).
* **الملفات المنجزة**:
  * جدول المقارنة الكامل: [`T16_Alternative_GT.csv`](../results/phase2/T16_Alternative_GT.csv)
  * جدول ملخص الذكاء الاصطناعي فقط: [`T16_Alternative_GT_AIonly.csv`](../results/phase2/T16_Alternative_GT_AIonly.csv)
  * كود التنفيذ: [`task1_alt_gt.py`](../src/phase2/task1_alt_gt.py)

---

### ✅ المهمة 2 — توثيق مجموعة الأخبار الثانية (CNN/DailyMail) ونتائجها
* **المطلوب**: توثيق النماذج، البرومبتات، المعلمات، ومصدر البيانات، مع إرفاق نتائج التقييم.
* **النتائج والتوثيق الكامل**:
  1. **مصدر النصوص البشرية**: مجموعة Hugging Face `abisee/cnn_dailymail` (الإصدار 3.0.0، test split، seed=42، 1000 مقال) عبر السكربت [`prepare_cnn_dailymail.py`](../tools/second_corpus/prepare_cnn_dailymail.py). النصوص تبدأ بـ `(CNN)`، واستخدم ملخص `highlights` كـ `Prompt` والمقال كـ `Human_story`. (كانت تسمى "BBC" خطأً في ملفات قديمة — أُعيدت تسمية كل الأكواد والبيانات والجداول إلى CNN/DailyMail).
  2. **سكربت وإعدادات التوليد**: سكربت [`generate_from_apis.py`](../tools/second_corpus/generate_from_apis.py).
     * قالب البرومبت:
       > *"Write a complete, professional news article based on the following summary/abstract. Do not add any commentary or meta-text — only output the article itself.\n\nSummary:\n{prompt}"*
     * المعلمات: `temperature: 0.7`, `max_tokens: 1024`.
     * النماذج المستخدمة:
       * `gemma_2_9b`: OpenRouter `google/gemma-3-12b-it`
       * `qwen_2_72b`: OpenRouter `qwen/qwen-2.5-72b-instruct`
       * `llama_3_8b`: OpenRouter `meta-llama/llama-3.1-8b-instruct`
       * `mistral_7b`: OpenRouter `mistralai/mistral-nemo`
       * `deepseek_pro`: OpenRouter `deepseek/deepseek-v4-pro` (بديل لـ Yi-Large)
       * `claude_haiku`: OpenRouter `anthropic/claude-3-haiku:beta`
       * `gpt-4o-mini`: OpenAI API `gpt-4o-mini`
     * الملف المولد: [`results/extra/task2/generated_full1000.csv`](../results/extra/task2/generated_full1000.csv) (1000 صف × 9 أعمدة).
  3. **نتائج تقييم النموذج الهجين (Zero-Shot)**: تم استخراج الخصائص عبر [`helper/cross_corpus_feature_eval.py`](../helper/cross_corpus_feature_eval.py):
     * جدول النتائج: [`results/tables/T14_Second_Corpus_Validation.csv`](../results/tables/T14_Second_Corpus_Validation.csv):
       * الثنائي (Binary): الدقة **84.66%**، Macro-F1: **70.14%**.
       * المتعدد (Multi-Class): الدقة **26.56%**، Macro-F1: **19.92%**.
     * مصفوفة الالتباس: [`results/figures/F15_Second_Corpus_Confusion_Matrix.png`](../results/figures/F15_Second_Corpus_Confusion_Matrix.png).
* **الملف التوثيقي التفصيلي**: [`docs/06_cnn_provenance.md`](06_cnn_provenance.md).

---

### ✅ المهمة 3 — مصفوفة التباس RoBERTa تحت هجوم Gemma
* **المطلوب**: المصفوفة الرقمية متعددة الأصناف.
* **ما تم تنفيذه**: إعادة الاستدلال بنموذج `RoBERTa_Multi` المحفوظ على مجموعة Gemma-27B (2072 مستنداً، 17.9 مللي ثانية/مستند على RTX 3060).
* **النتيجة**: الدقة 16.84%، و Macro-F1 بلغ 9.80%. التنبؤات تنهار نحو Gemma-2-9B بنسبة 75.5% (مما يثبت استنتاج المهمة 1).
* **الملفات المنجزة**:
  * الأعداد الخام: [`T17_RoBERTa_Gemma_CM_counts.csv`](../results/phase2/T17_RoBERTa_Gemma_CM_counts.csv)
  * النسب المعيارية للأسطر: [`T17_RoBERTa_Gemma_CM_rownorm.csv`](../results/phase2/T17_RoBERTa_Gemma_CM_rownorm.csv)
  * التنبؤات الكاملة: [`T17_RoBERTa_Gemma_preds.csv`](../results/phase2/T17_RoBERTa_Gemma_preds.csv)
  * جدول الملخص: [`T17_RoBERTa_Gemma_summary.csv`](../results/phase2/T17_RoBERTa_Gemma_summary.csv)
  * الشكل البياني: [`F17_RoBERTa_Gemma_CM.png`](../results/phase2/F17_RoBERTa_Gemma_CM.png)
  * كود التنفيذ: [`task3_roberta_cm.py`](../src/phase2/task3_roberta_cm.py)

---

### ✅ المهمة 4 — مقدار التشويه (BERTScore و Self-BLEU)
* **المطلوب**: قياس التشويه الدلالي واللفظي بين الأصل وإعادة الصياغة لكل نموذج.
* **ما تم تنفيذه**: حساب BERTScore-F1 (نموذج `roberta-large`) و sentence-BLEU على 2100 زوج لكل نموذج (6 نماذج — أُضيف Gemma-27B بإعادة بناء أزواجه عبر مطابقة TF-IDF ضمن نفس الكاتب ضد `data/need_processing/articles.csv`، N=2094).
* **النتيجة**: عائلة Gemma الأشد إعادة صياغة (Gemma-27B: BLEU = 10.72، Gemma-2-9B: BLEU = 10.86)، بينما Qwen-2 الأكثر تحفظاً وحفاظاً على النص الأصلي (BLEU = 51.5). وهذا يفسر هندسياً ولغوياً لماذا ينفرد Gemma بمحو البصمة وإعادة كتابتها.
* **الملفات المنجزة**:
  * جدول النتائج: [`T19_Perturbation_Magnitude.csv`](../results/phase2/T19_Perturbation_Magnitude.csv) (6 صفوف)
  * أزواج Gemma-27B المعاد بناؤها: [`data/processed/raw_paraphrased_gemma27b.csv`](../data/processed/raw_paraphrased_gemma27b.csv)
  * كود التنفيذ: [`task4_perturb.py`](../src/phase2/task4_perturb.py)، [`task4b_gemma27b_match.py`](../src/phase2/task4b_gemma27b_match.py)، [`task4c_gemma27b_scores.py`](../src/phase2/task4c_gemma27b_scores.py)

---

### ✅ المهمة 5 — توزيع الأصناف المتوقعة تحت الضجيج والترجمة
* **المطلوب**: حساب مجاميع أعمدة مصفوفة الالتباس تحت أخطاء 5% و20% والترجمة العكسية بـ NLLB.
* **ما تم تنفيذه**: إعادة الاستدلال بالنموذج الهجين واستخراج توزيع التوقعات.
* **النتيجة**: عند نسبة أخطاء 20%، ينهار النموذج بالكامل نحو صنف **Human بنسبة 90.9%** (Human attractor). أما الترجمة العكسية فيبقى التوزيع متوازناً (~14% لكل صنف) دون جاذب أحادي.
* **الملفات المنجزة**:
  * توزيع أخطاء 5%: [`T20_PredDist_Misspell_5%.csv`](../results/phase2/T20_PredDist_Misspell_5%25.csv)
  * توزيع أخطاء 20%: [`T20_PredDist_Misspell_20%.csv`](../results/phase2/T20_PredDist_Misspell_20%25.csv)
  * توزيع الترجمة: [`T20_PredDist_Translation_NLLB.csv`](../results/phase2/T20_PredDist_Translation_NLLB.csv)
  * كود التنفيذ: [`task5_preddist.py`](../src/phase2/task5_preddist.py)

---

### ✅ المهمة 6 — إعادة تدريب النموذج الهجين بدون Gemma (استئصال الصنف)
* **المطلوب**: استبعاد Gemma-2-9B من التدريب (6 أصناف) وتقييم الصمود على Gemma-27B، Mistral، وCNN.
* **ما تم تنفيذه**: تدريب نموذج XGBoost جديد على 6 أصناف مع الحفاظ على النماذج الأساسية.
* **النتيجة**: بدون وجود صنف Gemma في فضاء التصنيف، لا تعود العينات إلى صنفها الأصلي؛ بل تنقسم كتلة عينات Gemma وتنهار بالتساوي بين Human و GPT-4o (في مجموعة Gemma-27B: ذهبت 579 عينة إلى Human و 529 إلى GPT-4o).
* **الملفات المنجزة**:
  * الملخص: [`T21_NoGemma_summary.csv`](../results/phase2/T21_NoGemma_summary.csv)
  * مصفوفات الالتباس: [`T21_CM_CNN.csv`](../results/phase2/T21_CM_CNN.csv)، [`T21_CM_Gemma-27B.csv`](../results/phase2/T21_CM_Gemma-27B.csv)، [`T21_CM_Mistral-7B.csv`](../results/phase2/T21_CM_Mistral-7B.csv)
  * الرسوم البيانية: [`F21_CM_CNN.png`](../results/phase2/F21_CM_CNN.png)، [`F21_CM_Gemma-27B.png`](../results/phase2/F21_CM_Gemma-27B.png)، [`F21_CM_Mistral-7B.png`](../results/phase2/F21_CM_Mistral-7B.png)
  * كود التنفيذ: [`task6_nogemma_eval.py`](../src/phase2/task6_nogemma_eval.py)

---

### ✅ المهمة 7 — بروتوكول RoBERTa بخمس بذور عشوائية
* **المطلوب**: تدريب وتقييم RoBERTa الثنائي والمتعدد عبر 5 بذور (42، 123، 999، 2024، 8080) وحساب المتوسط والانحراف المعياري (mean ± std).
* **ما تم تنفيذه**: تدريب النماذج عبر السكربت [`run_seed.py`](../src/phase2/run_seed.py) واستخراج الإحصاءات التجميعية.
* **النتيجة**:
  * الثنائي مستقر جداً (Clean: دقة 0.9973 ± 0.0013).
  * المتعدد نظيف بدقة 0.9699 ± 0.0105، لكنه ينهار تحت هجوم Gemma بدقة 0.1847 ± 0.0154 عبر كل البذور دون استثناء.
* **الملفات المنجزة**:
  * ملخص الإحصاءات: [`T25_5seed_summary.csv`](../results/phase2/T25_5seed_summary.csv)
  * كافة القياسات المفصلة: [`T25_all_seeds.csv`](../results/phase2/T25_all_seeds.csv)
  * نتائج كل بذرة: [`T25_seed123.csv`](../results/phase2/T25_seed123.csv)، [`T25_seed999.csv`](../results/phase2/T25_seed999.csv)، [`T25_seed2024.csv`](../results/phase2/T25_seed2024.csv)، [`T25_seed8080.csv`](../results/phase2/T25_seed8080.csv)

---

### ✅ المهمة 8 — مسافات المراكز في فضاء الخصائص (Centroids)
* **المطلوب**: قياس المسافة الإقليدية والـ Cosine بين مراكز التوزيعات الأصلية والمُعاد صياغتها.
* **النتيجة**: جميع مجموعات إعادة الصياغة أقرب هندسياً إلى منطقة Yi-Large و Gemma-2-9B و Mistral-7B، ولا تقترب هندسياً من مركز كاتبها الأصلي (ترتيب جيب التمام للمؤلف الأصلي بين 2 إلى 5)، مما يقدم إثباتاً هندسياً على الانجذاب المركزي.
* **الملفات المنجزة**:
  * المسافات الإقليدية: [`T22_Centroid_Euclidean.csv`](../results/phase2/T22_Centroid_Euclidean.csv)
  * تشابه جيب التمام: [`T22_Centroid_Cosine.csv`](../results/phase2/T22_Centroid_Cosine.csv)
  * كود التنفيذ: [`task8_centroids.py`](../src/phase2/task8_centroids.py)

---

### ✅ المهمة 9 — تفكيك زمن الاستدلال (Latency Breakdown)
* **المطلوب**: قياس زمن الاستدلال لكل مرحلة ومقارنة النموذج الهجين بنموذج RoBERTa.
* **النتيجة**: يستغرق النموذج الهجين حوالي **8.2 ثانية/مستند** (BERT: 4.0 ثانية + Semantic: 1.0 ثانية + باقي الخصائص وXGBoost)، بينما يستغرق RoBERTa قرابة **18 مللي ثانية/مستند** (أي أن الهجين أبطأ بـ ~450 ضعفاً عند المعالجة الفردية Batch=1).
* **الملفات المنجزة**:
  * جدول أزمنة المكونات: [`T24_Latency.csv`](../results/phase2/T24_Latency.csv)
  * كود القياس: [`task9_latency.py`](../src/phase2/task9_latency.py)

---

### ✅ المهمة 10 — تحليل الثقة والمعايرة (Confidence Analysis)
* **المطلوب**: دراسة متوسطات ثقة التنبؤ للحالات الصحيحة والخاطئة.
* **النتيجة**: النموذج يظهر إفراطاً في الثقة (Overconfidence) على مجموعات Gemma بمتوسط ثقة ~0.90 للتنبؤات الصحيحة والخاطئة على حد سواء (غياب المعايرة)، بينما يُظهر تمييزاً أفضل في الثقة على Qwen و GPT-4o-mini.
* **الملفات المنجزة**:
  * جدول الثقة: [`T23_Confidence.csv`](../results/phase2/T23_Confidence.csv)
  * كود التنفيذ: [`task10_confidence.py`](../src/phase2/task10_confidence.py)

---

## الخلاصة النهائية

جميع المهام العشر منجزة وموثقة بشكل قاطع، وكافة السكربتات، الجداول، الرسوم البيانية، ونماذج التقييم محفوظة داخل المستودع وجاهزة لإرفاقها للأستاذ المشرف ومجلة Q1.
