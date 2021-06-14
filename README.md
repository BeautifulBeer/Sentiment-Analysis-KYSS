# KYSS
Sentiment analysis project, which is conducted during AIGS538, POSTECH.

## Training Notebooks & scripts
You should change the configs("EDIT" part) in the code in the python files.

### Active Dropout (only at test time)
- RoBERT/robert_colab_ver_active_dropout.py
### Big Transfer
Finetune RoBERT model using AB dataset. First finetuning was done on the RK dataset, and second finetuning was done on the K dataset. Active dropout was applied in the both finetuing process.
- RoBERT/robert_colab_ver_bigtransfer.py: Trained on Books_5_preprocessed_trimmed.csv
- RoBERT/robert_colab_ver_bigtransfer_finetune.py: Finetune with train_plus.csv/train_final.csv
- RoBERT/robert_colab_ver_bigtransfer_finetune_active_dropout.py: Finetune with train_plus.csv/train_final.csv. Apply active dropout when finetuning
### Domain Adpatation
Finetune RoBERT model using AM dataset. First finetuning was done on the RK dataset, and second finetuning was done on the K dataset. Active dropout was applied in the both finetuing process.
- RoBERT/robert_colab_ver_data.py: Trained on merged_250742_train_plus.csv
- RoBERT/robert_colab_ver_data_finetune.py: Finetune with train_plus.csv/train_final.csv
- RoBERT/robert_colab_ver_data_finetune_active_dropout.py: Finetune with train_plus.csv/train_final.csv. Apply active dropout when finetuning
### Domain Randomization
Finetune RoBERT model using AA dataset. First finetuning was done on the RK dataset, and second finetuning was done on the K dataset. Active dropout was applied in the both finetuing process.
- RoBERT/robert_colab_ver_domain.py: Trained on merged_10000_train_plus.csv
- RoBERT/robert_colab_ver_domain_finetune.py: Finetune with train_plus.csv/train_final.csv
- RoBERT/robert_colab_ver_domain_finetune_active_dropout.py: Finetune with train_plus.csv/train_final.csv. Apply active dropout when finetuning

### Multi-Task Learning

Train a single RoBERTa model for several similar tasks such as SST-5, SST-2 etc. After training step, to focus on the real domain(Kaggle competition), fine-tuning is applied using the K dataset. Active dropout is used to improve generalization performance.

- Multi_Task/RoBERTa_Multi_Task.ipynb : RK, TM, RS, AM are for multi-task learning, K is for fine-tuning

---

## Method
- RoBERT
  - A pre-trained model developed from BERT
- Optimizer
  - AdamW
- Regularization
  - Label Smoothing (LS)
  - Multi-task Learning (MTL)
  - Active dropout (AD)
- Extended Dataset
  - Domain adaptation (DA)
  - Domain randomization[4] (DR)
  - Big Transfer Learning[5] (BT)
- Additional processing
  - Finetuning (FTn)
  - Knowledge Distilation[6] (KDn)

## Dataset
- K ([train_final.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EUe_peW3jfVFrS7FJfw3OVEBMS_g_3_vbBj4JEcYL37FxQ?e=rlxIWJ)): Kaggle Dataset (11,543)
- RK ([train_plus.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/ER9fyGMMtCpKqaela7WJQgsBkplRZDw0pA-H5rNiB_F0bQ?e=Ss81N9)): Kaggle dataset + Rotten Tomato Movie Review (75,719)
- AB ([Books_5_preprocessed_trimmed.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EWYNLcLNVK1Cs-hVUtMYqi0B2wyxUBdhYYrD6GqqIShg-A?e=4R2eQ3)): Amazon Review Data 2018 (5-core Books; 13,205,248)
- AA ([merged_10000_train_plus.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EZ6jg46Q5udMslObQ2Z-8REBVHdClAv1xQssv-x447GwvQ?e=Tsf6gh)): Amazon Review Data 2018 (5-core all categories upto 10K/cat; 250,742) + Kaggle Dataset
- AM ([merged_250742_train_plus.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EXEVyLcF5UxJhoKPgIqbo4sBKPN6nmhwkTZr9oozS55-bw?e=V7uIcA)): Amazon Review Data 2018 (5-core Movie&TV; 250,742) + Kaggle Dataset
- TM ([train_IM.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EShwz4bT9yRPl8i0O82Ir0wBAffk-aOU9fRj1L4u5e2yiA?e=eCYgUD)): IMDB Movie Review (50,000)
- RS ([train_sarcasm.csv](https://postechackr-my.sharepoint.com/:x:/g/personal/ywshin_postech_ac_kr/EUEeGjRtw_1BsxqmqvOhaYkBRpPSMDPDZoi-jvjkNmVWOA?e=eiXbPZ)): Reddit-Sarcasm (100,000)

## Experiment & Results
![results](https://github.com/BeautifulBeer/kyss/blob/develop/results/result.png)

## Conclusion
The generalization is key issue in NLP task. In this project, active dropout and multi-task learning is applied to regularize the model and the effect is transparent, it improves accuracy compared to the baseline model. In addition, the extended dataset and related techniques, domain adaptation, domain randomization and big transfer learning also effective to increase accuracy. Standalone application of each method is effective, but, additional improvement is obtained with combined application with it.

## References
[1] Yarin Gal(2016), “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning”, Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:1050-1059.

[2] Xiaodong Liu (2019), “Multi-Task Deep Neural Networks for Natural Language Understanding”, ACL

[3] Liu, Yinhan (2019), "RoBERTa: A robustly optimized bert pretraining approach", arXiv preprint: 1907. 11692.

[4] Josh Tobin(2017), Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

[5] Alexander Kolesnikov(2019), Big Transfer (BiT): General Visual Representation Learning

[6] Tommaso Furlanello, et al.(2018), Born Again Neural Networks
