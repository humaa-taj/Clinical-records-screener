---
license: other
license_name: public-domain
license_link: LICENSE
dataset_info:
  features:
  - name: annotation_id
    dtype: int64
  - name: patient_id
    dtype: string
  - name: note
    dtype: string
  - name: trial_id
    dtype: string
  - name: trial_title
    dtype: string
  - name: criterion_type
    dtype: string
  - name: criterion_text
    dtype: string
  - name: gpt4_explanation
    dtype: string
  - name: explanation_correctness
    dtype: string
  - name: gpt4_sentences
    dtype: string
  - name: expert_sentences
    dtype: string
  - name: gpt4_eligibility
    dtype: string
  - name: expert_eligibility
    dtype: string
  - name: training
    dtype: bool
  splits:
  - name: train
    num_bytes: 1145187
    num_examples: 1015
  download_size: 176088
  dataset_size: 1145187
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
