import mne

bdf_path = r"F:\DiTing_BrainCar-main\demos\brainflow_demos\EEG_20260708_16_23_51.bdf"

raw = mne.io.read_raw_bdf(bdf_path, preload=True, stim_channel="Trigger/Status", verbose=False)
events = mne.find_events(raw, stim_channel="Trigger/Status", shortest_event=1, mask=255, mask_type='and', verbose=False)
print("Channel No.\tChannel Name")
for ch_no, ch_name in enumerate(raw.ch_names, start=1):
    print(f"{ch_no}\t{ch_name}")
print()

print(events)
