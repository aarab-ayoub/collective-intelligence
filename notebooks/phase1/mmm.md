Great questions — let me break all of this down clearly.

**What is an `.npz` file?**

It's just a zip archive of multiple numpy arrays bundled into one file. When your preprocessing notebook ran, it took all those raw PTB-XL `.dat` records, processed them, and packed the results into one neat file. Think of it like this:

```
ptbxl_mit_style_dataset.npz
  ├── X_train  → array of shape (N, 187)   ← the actual signal values
  ├── y_train  → array of shape (N,)       ← the label for each sample
  ├── X_val    → ...
  ├── y_val    → ...
  ├── X_test   → ...
  ├── y_test   → ...
  └── label_names → ["CD", "HYP", "MI", "NORM", "STTC"]
```

So yes, it still contains all those samples — tens of thousands of rows — just stored as binary numpy arrays instead of CSV text. It loads instantly and takes way less disk space. The line `np.load(NPZ_PATH)` unpacks it back into arrays.

---

**What is a "beat"?**

A raw PTB-XL record is a 10-second ECG recording across 12 leads. That full 10-second signal is noisy, long, and hard for a CNN to learn from directly. A **beat** is a single heartbeat — one PQRST wave — extracted from that signal.

Your preprocessing notebook did this:
1. Took Lead II from the 10-second record
2. Found the R-peaks (the tall spikes) using `find_peaks`
3. Cut a 187-sample window around each R-peak (60 samples before, 127 after)
4. Normalised each window to [0, 1]
5. Called that one training sample

So one PTB-XL record → up to 40 individual beat samples. That's why you end up with tens of thousands of samples from only ~21k records.**What made notebook 02 work — the key insight**

The core reason the MIT-style approach works better than the original PTB-XL pipeline is that the learning problem became dramatically simpler. Instead of feeding the model a 10-second, 12-lead signal and asking it to figure out what's happening globally, you give it one isolated heartbeat — 187 numbers, one lead — and ask "is this normal or not?" The signal-to-noise ratio is incomparably better.

The specific things that made it healthy:

The per-beat min-max normalisation was critical. Each heartbeat gets independently scaled to [0, 1] regardless of the patient's overall voltage. This means the model learns the *shape* of a heartbeat, not the amplitude. A patient with a weak signal and a patient with a strong signal produce identically-scaled beats — the model only has to recognise the morphology (does this PQRST wave look abnormal?).

The record-level split before beat extraction prevented the most common form of data leakage. If you split by beat, the same patient's heartbeat 7 could end up in train and heartbeat 8 in val — the model just memorises patient-specific quirks and gets artificially high accuracy. Splitting by record first means every beat from one patient stays in the same split.

The `MAX_BEATS_PER_RECORD = 40` cap prevented dominant records from flooding the dataset. Without it, a patient with clear arrhythmia and many detectable peaks could contribute hundreds of beats and bias the model toward their particular morphology.

**What are beats and labels concretely**

A beat is literally one heartbeat — one PQRST complex. The P wave is atrial contraction, the tall R spike is ventricular contraction, the T wave is recovery. Your preprocessing found each R spike in Lead II and cut a 187-sample window centred slightly after it (60 samples before, 127 after — biased toward the recovery phase). Each window gets min-max scaled and becomes one row in `X_train`.

The label comes from the original PTB-XL metadata. Each record was annotated by cardiologists with SCP diagnostic codes, which map to five superclasses: NORM (normal), MI (myocardial infarction), STTC (ST/T changes), CD (conduction disturbance), HYP (hypertrophy). In binary mode, NORM stays 0 and everything else collapses to 1 (ABN). Every beat extracted from a NORM record gets label 0, every beat from any other record gets label 1.

**What is the actual goal of this project**

At the module level (IoT embedded inference), the goal is to train a model on your host machine and then make it run efficiently on very constrained devices — the three VMs with 500 MB, 1 GB, and 2 GB RAM. The ECG classification task is the vehicle for that. But the ECG classification itself is clinically meaningful: given a single heartbeat segment, can a lightweight model automatically flag whether the heart looks normal or abnormal? That's exactly what a wearable ECG device would need to do in real time — detect beat by beat, send an alert if something looks wrong.