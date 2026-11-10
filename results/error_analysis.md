# Error analysis

Scope: the fine-tuned Qwen2.5-1.5B model (run R3, tag `final-config`) on the held-out test split, 206 criteria from 8 patients. Source: `results/eval/after_test/predictions.jsonl`. This analysis was done after the final evaluation, and nothing in the model, prompt or settings changed because of it.

The cause given for each case is a reading of the note, criterion and output. It is a judgment about likely cause, and it has not been confirmed by a clinician or by inspecting model internals.

## Overall pattern

The model got 150 of 206 labels right, so 56 were wrong. Another 27 rows had the right label with a different evidence set from the gold annotation. No output was invalid, and no citation pointed at a sentence that does not exist.

| Gold label | Predicted label | Rows |
|---|---|---|
| not enough information | not excluded | 24 |
| not applicable | not excluded | 8 |
| included | not enough information | 7 |
| not included | not enough information | 6 |
| not applicable | not enough information | 4 |
| not excluded | not enough information | 3 |
| not enough information | not applicable | 2 |
| included | not included | 1 |
| not excluded | not applicable | 1 |

Two habits explain most of this.

- **On exclusion criteria the model defaults to "not excluded".** That accounts for 32 of the 56 errors. In the training data 74.7% of "not excluded" rows have no cited sentence, so the model learned that a silent note usually means "not excluded".
- **On inclusion criteria it retreats to "not enough information".** It does this when the answer needs a clinical inference. "Not enough information" had recall 1.0 on inclusion rows, while "not included" had recall 0.14 (1 of 7).

The largest cluster is one patient and one trial: 17 of the 24 rows in the top line come from patient sigir-201421 on trial NCT00997100. With only 8 test patients, a single annotator's habit on one long trial moves the totals. The patient-level bootstrap interval for macro-F1 is 0.43 to 0.64. The "excluded" label has no test examples at all, so its performance could not be measured on this split.

## Cases

Evidence lists are sentence IDs in the numbered note.

### 1. Inclusion polarity flip on a negated criterion (ID 176)
- **Patient, trial:** sigir-201410, NCT02264964
- **Criterion:** inclusion, "No medical history of central vena catheterization."
- **Note:** [0] says she had cardiac catheterization via the right femoral artery.
- **Gold:** included, evidence [0]. **Predicted:** not included, evidence [].
- **Cause:** the criterion is negated and the note has a near match. An arterial catheter is not a central venous catheter, but the shared word "catheterization" appears to have triggered a failure verdict. This is the only included/not-included flip on the test split.
- **App behavior:** with no citation the row is flagged and the patient goes to "Needs review", so it does not produce "Likely ineligible".

### 2. Missing information treated as "no" (ID 403)
- **Patient, trial:** sigir-201421, NCT00997100
- **Criterion:** exclusion, positive screening for hepatitis B, hepatitis C or HIV.
- **Note:** no sentence mentions viral screening.
- **Gold:** not enough information, evidence []. **Predicted:** not excluded, evidence [].
- **Cause:** the training default described above. In the same category, ID 169 (patient sigir-201410) has an eGFR threshold with no renal result in the note. Absence of a test result is not a negative result.
- **App behavior:** an uncited "not excluded" is shown as unverified, so a missing result is not presented as a pass.

### 3. Correct sentence, inference not made (ID 186)
- **Patient, trial:** sigir-201411, NCT01526382
- **Criterion:** inclusion, "Heart rate of at least 90/min".
- **Note:** [2] says she is "tachypneic and tachycardic". No number is given.
- **Gold:** included, evidence [2]. **Predicted:** not enough information, evidence [2].
- **Cause:** the model found the right sentence but did not connect "tachycardic" to a rate above the threshold. The evidence was right and the label was too cautious.

### 4. Scale conversion not made (ID 195)
- **Patient, trial:** sigir-201411, NCT01723137
- **Criterion:** inclusion, "Reported pain greater than or equal to 3 out of 10".
- **Note:** [0] describes "excruciating pain".
- **Gold:** included, evidence [0]. **Predicted:** not enough information, evidence [0].
- **Cause:** a verbal description had to be mapped to a numeric scale. Same pattern as case 3.

### 5. Diagnosis had to be inferred from vital signs (ID 185)
- **Patient, trial:** sigir-201411, NCT01526382
- **Criterion:** inclusion, diagnosed with shock, ARDS, or both.
- **Note:** [2] tachypneic and tachycardic, [3] blood pressure 80/60. The word "shock" never appears.
- **Gold:** included, evidence [2, 3]. **Predicted:** not enough information, evidence [].
- **Cause:** the annotator inferred shock from the vital signs. The model did not, and it also cited nothing.

### 6. Abbreviated condition (IDs 382 and 413)
- **Patient, trials:** sigir-201421, NCT00997100 and NCT01520155
- **Criteria:** ID 382, inclusion, "Fulfil at least 4 criteria for SLE as defined by the American College of Rheumatology". ID 413, inclusion, "Patients with systemic Lupus erythematosus".
- **Gold:** both included. ID 382 evidence [0, 1, 2]. ID 413 evidence [0, 1, 2, 3].
- **Predicted:** ID 382 not enough information, evidence [0, 1, 2, 3]. ID 413 included, evidence [0, 1, 2].
- **Cause:** for the same patient the spelled-out disease was accepted and the abbreviated form was not. This is suggestive only, since ID 382 also requires counting diagnostic criteria, which is harder in its own right.

### 7. Unsupported citation with the right label (ID 416)
- **Patient, trial:** sigir-201422, NCT00723788
- **Criterion:** inclusion, referred from the emergency department for suspected appendicitis.
- **Gold:** included, evidence [0, 5]. **Predicted:** included, evidence [0, 1, 3].
- **Cause:** sentence [3] is about sexual history and menses, which has no bearing on the criterion, and the ultrasound in [5] was missed. A correct label can still come with a wrong justification, so label accuracy alone overstates quality.

### 8. Blanket citation (ID 832)
- **Patient, trial:** sigir-201516, NCT00839124
- **Criterion:** inclusion, symptom score no greater than 6 of 24.
- **Gold:** not enough information, evidence []. **Predicted:** not enough information, evidence [0, 1, 2, 3, 4, 5, 6, 7].
- **Cause:** no symptom score appears anywhere in the note, yet the model cited every clinical sentence. The verifier confirms that cited IDs exist, but it cannot judge whether they are relevant.

### 9. Right label with evidence omitted (ID 414)
- **Patient, trial:** sigir-201421, NCT01520155
- **Criterion:** exclusion, "Patients without systemic Lupus erythematosus".
- **Gold:** not excluded, evidence [0, 1, 2, 3]. **Predicted:** not excluded, evidence [].
- **Cause:** the training habit of leaving "not excluded" uncited. This row costs evidence recall even though the label is right.

### 10. Not-applicable confusion and inconsistent gold labels (IDs 697 and 322)
- **ID 697:** sigir-20158, NCT00393913, exclusion, "Pregnant". The patient is a 10-year-old boy. **Gold:** not applicable, evidence []. **Predicted:** not excluded, evidence [].
- **ID 322:** sigir-201417, NCT01745731, exclusion, pregnant women or women of childbearing age without adequate contraception. The patient is a 48-year-old man. **Gold:** not excluded, evidence [0]. **Predicted:** not applicable, evidence [].
- **Cause:** both are the same kind of criterion applied to a male patient, and the gold labels differ. The model made the opposite choice each time. Part of the weak "not applicable" score (F1 0.29) reflects a boundary that the annotations do not draw consistently.

### 11. Temporal reasoning and negation (ID 836)
- **Patient, trial:** sigir-201516, NCT00839124
- **Criterion:** inclusion, history of wheezing or physician-diagnosed asthma after age 6.
- **Note:** [0] the patient is 4 years old. [1] "no history of wheezing".
- **Gold:** not included, evidence [0, 1]. **Predicted:** not enough information, evidence [1].
- **Cause:** the model did not use the patient's age to rule out an event after age 6, and it did not treat the negated history as settling the criterion.

### 12. Facts that appear to conflict (ID 827)
- **Patient, trial:** sigir-201516, NCT00930826
- **Criterion:** inclusion, clinical diagnosis of bronchial asthma.
- **Note:** [1] "no history of wheezing", while [3] and [6] report current wheezing.
- **Gold:** not enough information, evidence [0, 1, 2, 3, 6]. **Predicted:** not enough information, evidence [1, 3, 4].
- **Cause:** the label is right but only part of the evidence matches. The note is not truly contradictory, since the wheezing is new. No test note contained a real contradiction, so this category is only lightly tested.

## Invalid output

No invalid output occurred on the validation or test splits (412 rows). The verifier's handling of invalid JSON, illegal labels and non-existent sentence IDs is covered by `tests/test_verifier.py`. It could not be illustrated with a real failure from these runs.

## Rehearsal on invented notes

This is separate from the test split and is not a benchmark. Five invented notes and 24 criteria were run through the finished app, after the model was frozen. The expected labels were written by the developer and have not been reviewed by a clinician. The full output is in `results/rehearsal_unseen_notes.txt`. The model matched 14 of the 24 expected labels, which is below its test accuracy and is too small a sample to compare with it. Nothing was changed in response.

- **Numeric thresholds are unreliable.** Of 11 criteria that needed a number or range compared, 6 were right and 5 wrong. With a BMI of 33, "BMI of at least 30" came back "not included" while citing the correct sentence. HbA1c of 7.9 against "7.0 to 9.5" came back "not enough information". The app's own built-in example shows the same fault, marking HbA1c of 8.4 as "not included" for "between 7 and 10 percent".
- **A cited failure can still be wrong.** The BMI error above made that patient "Likely ineligible". The overall rule requires a citation, but a citation shows where the model looked and does not prove its reasoning was right.
- **A stated exclusion can be missed.** "Current anticoagulant therapy" with apixaban in the note came back "not excluded" while citing that very sentence, and it counted as met. "History of tuberculosis" with treated tuberculosis in the note came back "not excluded" with no citation, which the app does flag for review.
- **Silence again became "not excluded".** Hepatitis B, rituximab and active cancer were not mentioned in their notes and all came back "not excluded" with no citation, matching the main test-split pattern. All three are flagged.
- **Headings can pull citations off target.** In a list-formatted note, "Atrial fibrillation" cited the heading line "History:" rather than the bullet below it. The app now removes heading lines and list marks from unseen notes before numbering them.

### Deterministic number check added to the application

Because the model cannot be changed after the test split was opened, a plain-code check now runs on its output in the application (`app/number_check.py`). When a criterion is a bare numeric comparison and the note holds one clear value, the code does the comparison and replaces a disagreeing verdict. Those rows carry the line "Adjusted by number check." and the model's original answer stays in the JSON report.

It was developed on training notes only (16 decisions, all matching the expert labels) and then judged once on the validation split against a bar fixed in advance: every decision had to match the expert label. It made 5 decisions, all correct, on 206 validation rows. All 5 were age criteria the model had already answered correctly, so it changed nothing on validation. The dataset gave no validation evidence for BMI or laboratory values, which are covered only by unit tests.

It was never run on the test split, and all reported test metrics are for the model alone. Rerunning the rehearsal with the check and the heading cleanup raised the match from 14 to 18 of 24 (`results/rehearsal_with_number_check.txt`). That figure is not independent evidence, because the check was built after seeing those same failures.

## What this means

- The app's rule that uncited verdicts need review directly limits the most common error, because the rows in case 2 are all uncited.
- The app is a screening aid and its verdicts need a person to check them, including those that cite evidence.
- Citation checks catch IDs that do not exist. Irrelevant citations, as in cases 7 and 8, are not caught and need a person to read the highlighted sentences.
- Better results on this task would need more examples of rare labels and a consistent annotation rule for "not applicable". This was not attempted after the test split was opened.
