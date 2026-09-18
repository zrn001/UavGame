import mne

raw = mne.io.read_raw_bdf(r"D:\User_code\zhz\7-放大器40指令刺激和处理程序\移动\处理\25年杭州终验\China-Mobile处理\MetaBCI-master\Data\0521\SSVEP_1.bdf", preload=True)
events = mne.events_from_annotations(
            raw, event_id=lambda x: int(x), verbose=False)[0]
print(events)


