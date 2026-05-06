إليك القائمة المدمجة للتحقق منها قبل إطلاق الكود للعمل طوال الليل:📂 1. البنية التحتية وقابلية إعادة الإنتاج (Infrastructure & Reproducibility)هذه المتطلبات أساسية حسب الخطة:  [x] تثبيت البذور العشوائية (Fixed Seed): ضمان استخدام نفس الـ Seed وتقسيم 80/20 في جميع التجارب.  [x] تنظيم المخرجات (Folder Structure): إنشاء المجلدات المطلوبة: results/tables/ و results/figures/ و results/models/ و results/logs/.  [x] حفظ النماذج: حفظ النماذج بصيغة .joblib مع بيانات المعايرة (Scalers).  📊 2. جداول النتائج (The Tables - T1 to T10)هذه الجداول ستولدها الأكواد التي برمجناها وسيتم حفظها كملفات CSV:  [x] T1: نتائج الكشف الثنائي (Baseline - Binary) للميزات فقط مقابل الميزات+بيرت.  [x] T2: نتائج التصنيف متعدد الفئات (Baseline - Multi-class).  [x] T3: المقياس إف-1 لكل فئة (Per-class F1) لجميع الكتاب الـ 7.  [x] T4: تحليل تدهور الأخطاء الإملائية بنسب متعددة (5%, 10%, 15%, 20%).  [x] T5: تحليل حساسية طول النص (قصير، متوسط، طويل) مع حدود الكلمات وأفضل 5 ميزات.  [x] T6: تعميم النطاق المتقاطع (Cross-Domain) لمقالات Essay و WP ونموذج Claude.  [x] T7: ملخص الهجمات العدائية (إعادة الصياغة، الترجمة، الأخطاء الإملائية 20%).  [x] T8: دراسة استئصال الميزات (Feature Ablation) لتأثير إزالة كل مجموعة ميزات.  [x] T9 (إضافة الدكتور): اختبار الدلالة الإحصائية (المتوسط والانحراف المعياري لـ 5 بذور عشوائية).[x] T10 (إضافة الدكتور): المقياس إف-1 لكل فئة (Per-class F1) لهجمات إعادة الصياغة والترجمة والأخطاء الإملائية.📈 3. الرسومات البيانية (The Figures - F1 to F14)يجب حفظ هذه الرسومات بدقة عالية (300 DPI):  [x] F1 & F2: مصفوفات الارتباك (Confusion Matrices) للمهام الثنائية ومتعددة الفئات مع بيرت.  [x] F3, F4, F5, F6: رسومات أهمية الميزات (Feature Importance) لجميع الحالات.  [x] F7 & F8: منحنيات تدهور الدقة مع الأخطاء الإملائية (Degradation Curves).  [x] F9: رسم بياني شريطي مجمع لحساسية طول النص (Length Sensitivity).  [x] F10 & F11: مصفوفات الارتباك لبيانات Essay و WP و Reuters.  [x] F12: مصفوفة الارتباك لهجوم إعادة الصياغة متعدد الفئات (Paraphrasing CM).  [x] F13: رسم بياني شريطي يوضح انخفاض الدقة عند استئصال الميزات.  [x] F14 (إضافة الدكتور): رسم ثنائي الأبعاد (PCA) يوضح قدرة الميزات على فصل الفئات.⏳ 4. المهام المتبقية للغد (Pending for Tomorrow)[ ] اختبار النماذج الخارجية (External Baselines): اختبار البيانات على DetectGPT أو أداة مشابهة للمقارنة.

Based on Dr. Alaa's email and the attached "Basel Report - 25 Apr.pdf", here is your comprehensive checklist. I have organized it by priority based on his instructions.

### **Phase 1: The 8 Quick Fixes (Verify & Correct)**
Dr. Alaa noted these will not take too much time. You should address these in your code and tables:

* [cite_start][x] **Issue 1 (T7 & F12 - Paraphrasing Gap):** Verify that the paraphrased dataset is balanced across all 7 classes[cite: 133]. [cite_start]Verify that the binary accuracy for the paraphrasing task is computed correctly[cite: 133]. , there was a problem in naming conation 
* [cite_start][x] **Issue 2 (F2 & F12 - Relabeling):** Change the label from `"accounts/yi-01-ai/models/yi-large"` to the clean, short name `"Yi-Large"` for publication[cite: 133].
* [cite_start][ ] **Issue 3 (T5 - Column Names):** Rename the column headers `"Bin Features Acc"` and `"Bin Feat+BERT F1"` to ensure standard and consistent naming[cite: 133].
* [cite_start][x] **Issue 4 (T3 - Consistency):** Relabel the full path of Yi-Large to `"Yi-Large"`[cite: 133]. [cite_start]Ensure the column order in T3 perfectly matches the confusion matrix order in F2[cite: 133].
* [cite_start][x] **Issue 5 (T4 - Misspelling Anomaly):** Confirm that the 20% misspelling result, where features-only (28.07%) outperforms features+BERT (24.07%), is actually correct and not a typo[cite: 133]. , there was  a problem in 20% misspelling results since we all non Gemnini writers
* [cite_start][ ] **Issue 6 (T8 - Burstiness Ablation):** Run the ablation experiment (removing burstiness) twice using different random seeds[cite: 133]. [cite_start]This is to confirm that the slight improvement in binary accuracy is reproducible and not just random noise[cite: 133].
* [cite_start][ ] **Issue 7 (T5 - Length Boundaries):** Add the actual word-count boundaries for each length bin (e.g., short=50:200 words) to the CSV or report[cite: 133].
* [cite_start][ ] **Issue 8 (T5 - Top Features):** Add the top 5 features for each length bin, or provide this information as a supplementary table[cite: 133].

---

### **Phase 2: Immediate Enhancements (Do BEFORE Writing)**
You must run these new experiments before starting to draft the paper:

* [cite_start][ ] **Statistical Significance Testing:** Run the baseline experiment 5 times using different random seeds[cite: 137]. [cite_start]Report the mean and standard deviation for both accuracy and F1-score to prove result stability[cite: 137].
* [cite_start][ ] **External Baselines:** Test one or two publicly available detectors (like DetectGPT, Binoculars, or GLTR) on your exact same test set to provide a comparison point[cite: 140]. 

---

### **Phase 3: Later Enhancements (Do DURING Writing)**
Dr. Alaa mentioned these are easy and can be generated while you write or during his review:

* [cite_start][ ] **Per-Class F3 for Attacks:** Generate per-class F1 score tables for the misspelling attack (at least at 5% and 20%), cross-domain generalization, and paraphrasing[cite: 144]. [cite_start]These will be used for supplementary materials[cite: 145].
* [cite_start][ ] **PCA or t-SNE Visualization:** Create a 2D projection chart showing how the feature-space separates the classes, comparing the layout with and without BERT[cite: 147].

