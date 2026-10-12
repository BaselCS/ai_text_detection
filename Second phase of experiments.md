Second phase of experiments  
**Priority 1 (no new training needed)**  
1. We need an alternative ground truth. This can be done by relabelling each paraphrased document with its paraphraser as the author and recompute accuracy and macro F1, for all six paraphrasers. Map each paraphraser to its nearest training class (Gemma-27B and Gemma-2-9B to Gemma-2-9B, Qwen to Qwen-2-72B, Mistral to Mistral-7B, LLaMA-3 to Llama-8B, GPT-4o-mini to GPT-4o).  
2. Tell me where the six AI versions of the BBC articles came from. If you generated them, send the models, prompts, and settings used.  
3. RoBERTa attribution confusion matrix under the Gemma paraphrase (you already have the predictions).  
**Priority 2**  
4. We need to rewrite the magnitude. So, for each paraphraser, compute BERTScore and self-BLEU between each original and its paraphrase, and report the mean per paraphraser.  
5. Predicted-class distribution (column totals of the confusion matrix) for the hybrid model under misspelling at 5% and 20%, and under translation.  
6. Retrain the hybrid attribution model with Gemma-2-9B removed from the label set (six classes), then re-run the Gemma-27B and Mistral paraphrase sets and the BBC set. Please send the confusion matrices.  
7. RoBERTa binary and multi-class across the five seeds (42, 123, 999, 2024, 8080).  
**Priority 3**  
8. Distance from the centroid of each paraphrased set to each training class centroid, in the Features + BERT space.  
9. Hybrid inference latency per document, measured the same way as the RoBERTa latency.  
10. Mean predicted confidence on correct versus wrong attributions, for the paraphrased sets.  
The last point I need to ask:   
Is the cross-domain WP set Writing Prompts or Wikipedia? and why do the Essay and WP counts show 396 and 398 rather than 400?  
Basel, please send Priority 1 first, as soon as it is ready. It important for me in order to decide the framing of the paper.  
Good Luck  
   
   
