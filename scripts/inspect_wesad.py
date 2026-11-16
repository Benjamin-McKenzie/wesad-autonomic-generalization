import pickle
import numpy as np

with open("data/raw/WESAD/S2/S2.pkl", "rb") as file:
    data = pickle.load(file, encoding = "latin1")

print(data.keys())

signal_data = data["signal"] # takes val from signal key and stores it as signal_data 
print(signal_data.keys()) # signal_data is nested data, "data" is still complete dict

chest_data = signal_data["chest"]
wrist_data = signal_data["wrist"]

print(chest_data.keys())
print(wrist_data.keys())

sampling_rates = {
    "chest": {
        "ACC": 700,
        "ECG": 700,
        "EMG": 700,
        "EDA": 700,
        "Temp": 700,
        "Resp": 700
    },
    "wrist": {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4
    }
}

# chest ECG
print("---------------Chest ECG---------------\n")
ch_ecg = chest_data["ECG"]
print("Shape: ", ch_ecg.shape)
print("Dim.", ch_ecg.ndim)
print("Total elements:", ch_ecg.size)
print("Data type:", ch_ecg.dtype)

print(ch_ecg[:5])
print()

# chest ACC
print("---------------Chest ACC---------------\n")
ch_acc = chest_data["ACC"]
print("Shape: ", ch_acc.shape)
print("Dim.", ch_acc.ndim)
print("Total elements:", ch_acc.size)
print("Data type:", ch_acc.dtype)

print(ch_acc[:5])
print()



# wrist ACC
print("---------------Wrist ACC---------------\n")
w_acc = wrist_data["ACC"]
print("Shape: ", w_acc.shape)
print("Dim.", w_acc.ndim)
print("Total elements:", w_acc.size)
print("Data type:", w_acc.dtype)

print(w_acc[:5],)
print()


# wrist BVP
print("---------------Wrist BVP---------------\n")
w_bvp = wrist_data["BVP"]
print("Shape: ", w_bvp.shape)
print("Dim.", w_bvp.ndim)
print("Total elements:", w_bvp.size)
print("Data type:", w_bvp.dtype)

print(w_bvp[:5],)
print()

print("Sampling frequencies (seconds): ")
#duration = observations/frequency

# chest ECG 700hz
print("Duration chest ECG: ", ch_ecg.shape[0]/sampling_rates["chest"]["ECG"])

# chest acc 700hz
print("Duration chest ACC: ", ch_acc.shape[0]/sampling_rates["chest"]["ACC"])

# wrist ACC sampling freq 32hz
print("Duration wrist ACC: ", w_acc.shape[0]/sampling_rates["wrist"]["ACC"])

# wrist BVP sampling freq 64hz
print("Duration wrist BVP: ", w_bvp.shape[0]/sampling_rates["wrist"]["BVP"])


# Labels Evaluation
print()
print("Labels Evaluation")
labels = data["label"]

print("Shape:", labels.shape)
print("Dimensions:", labels.ndim)
print("Data type:", labels.dtype)
print("First 20 labels:", labels[:20])

unique_labels, label_counts = np.unique(labels, return_counts = True) 
#np.unique for each distinct val in array
# return_counts = True returns how often each val appears

print("Unique labels:", unique_labels)
print("Label counts", label_counts)

print("Subject:", data["subject"])
print()

# Chest EMG
print("---------------Chest EMG---------------")
ch_emg = chest_data["EMG"]
print("Shape: ", ch_emg.shape)
print("Dim.", ch_emg.ndim)
print("Total elements:", ch_emg.size)
print("Data type:", ch_emg.dtype)
print("Duration chest EMG: ", ch_emg.shape[0]/sampling_rates["chest"]["EMG"])

print(ch_emg[:5])
print()

# Chest EDA
print("---------------Chest EDA---------------")
ch_eda = chest_data["EDA"]
print("Shape: ", ch_eda.shape)
print("Dim.", ch_eda.ndim)
print("Total elements:", ch_eda.size)
print("Data type:", ch_eda.dtype)
print("Duration chest EDA: ", ch_eda.shape[0]/sampling_rates["chest"]["EDA"])

print(ch_eda[:5])
print()

# Chest temp
print("---------------Chest Temp---------------")
ch_temp = chest_data["Temp"]
print("Shape: ", ch_temp.shape)
print("Dim.", ch_temp.ndim)
print("Total elements:", ch_temp.size)
print("Data type:", ch_temp.dtype)
print("Duration chest temp: ", ch_temp.shape[0]/sampling_rates["chest"]["Temp"])

print(ch_temp[:5])
print()

# Chest resp
print("---------------Chest Resp---------------")
ch_resp = chest_data["Resp"]
print("Shape: ", ch_resp.shape)
print("Dim.", ch_resp.ndim)
print("Total elements:", ch_resp.size)
print("Data type:", ch_resp.dtype)
print("Duration chest resp: ", ch_resp.shape[0]/sampling_rates["chest"]["Resp"])

print(ch_resp[:5])
print()

# Wrist EDA
print("---------------Wrist EDA---------------")
wr_eda = wrist_data["EDA"]
print("Shape: ", wr_eda.shape)
print("Dim.", wr_eda.ndim)
print("Total elements:", wr_eda.size)
print("Data type:", wr_eda.dtype)
print("Duration wrist EDA: ", wr_eda.shape[0]/sampling_rates["wrist"]["EDA"])
print(wr_eda[:5])
print()

# Wrist temp
print("---------------Wrist temp---------------")
wr_temp = wrist_data["TEMP"]
print("Shape: ", wr_temp.shape)
print("Dim.", wr_temp.ndim)
print("Total elements:", wr_temp.size)
print("Data type:", wr_temp.dtype)
print("Duration wrist temp: ", wr_temp.shape[0]/sampling_rates["wrist"]["TEMP"])

print(wr_temp[:5])
print()

# Summary Statistics
print("---------------Wrist bvp---------------")
print("Mean:", np.mean(w_bvp))
print("Standard Dev: ", np.std(w_bvp))
print("Min: ", np.min(w_bvp))
print("Max : ", np.max(w_bvp))
print("Range:", np.max(w_bvp) - np.min(w_bvp))

# first 10 seconds of w_bvp (64hz x 10 = 640)
bvp_segment = w_bvp[:640]

# small plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots() # creates figure and plotting area

ax.plot(bvp_segment) # draws the signal line

ax.set_title("Wrist BVP - First 10 Sec") # title
ax.set_xlabel("Observation") # horizontal axis label
ax.set_ylabel("BVP Amplitude") # vertical axis label

# plt.show() # display plot

# Check if S2 represents other 14 subjects
from pathlib import Path

data_dir = Path("data/raw/WESAD")

subject_files = sorted(data_dir.glob("S*/S*.pkl")) # searches subject_files for pickle files

print("# of subject files:", len(subject_files))

for file_path in subject_files:
    print(file_path)

# Check all subjects
import pickle

for file_path in subject_files:

    with open(file_path, "rb") as file:
        subject_data = pickle.load(file, encoding = "latin1")

    subject_id = subject_data["subject"]

    chest = subject_data["signal"]['chest']
    wrist = subject_data["signal"]["wrist"]
    labels = subject_data["label"]

    chest_ecg_duration = chest["ECG"].shape[0] / sampling_rates["chest"]["ECG"]
    wrist_bvp_duration = wrist["BVP"].shape[0] / sampling_rates["wrist"]["BVP"]
    wrist_eda_duration = wrist["EDA"].shape[0] / sampling_rates["wrist"]["EDA"]

    print(
        subject_id,
        "Chest ECG:", chest["ECG"].shape, # randomly selected metrics of various frequencies - 700hz
        "Duration:", chest_ecg_duration,
        "Wrist BVP:", wrist["BVP"].shape, # 64hz
        "Duration:", wrist_bvp_duration,
        "Wrist EDA:", wrist["EDA"].shape, # 4hz
        "Duration: ", wrist_eda_duration,
        "Labels:", labels.shape # 700hz
    )

