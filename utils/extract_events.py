# -*- coding: utf-8 -*-
"""
通用事件提取模块。

支持两种事件来源（自动适配）：
  1. 触发通道 —— 通过 mne.find_events(stim_channel=...) 从命名触发通道解析
  2. 文件标注 —— 通过 mne.events_from_annotations() 从 annotations 提取

用法：
  from utils.extract_events import extract_events

  # 方式①：BDF 文件有 "Trigger/Status" 触发通道
  events = extract_events(raw, trigger_ch_name="Trigger/Status")

  # 方式②：CNT/BDF 文件自带标注事件（原有逻辑）
  events = extract_events(raw, trigger_ch_name=None)
"""

import numpy as np
import mne


def extract_events(raw, trigger_ch_name=None, event_map=None):
    """
    统一的事件提取入口。

    参数
    ----
    raw : mne.io.Raw
        已加载的 MNE Raw 对象（需 preload=True）
    trigger_ch_name : str or None
        触发通道名称（如 "Trigger/Status"）。
        - 有值  → 使用 mne.find_events() 从该通道信号中解析事件
        - None  → 使用 mne.events_from_annotations() 从文件标注提取
    event_map : dict or None
        标注事件 ID 映射，仅在 trigger_ch_name=None 时使用。
        默认 {str(e): e for e in range(1, 255)}

    返回
    ----
    events : ndarray, shape (n_events, 3)
        MNE 格式 [sample_index, 0, label]
    """
    if trigger_ch_name is not None:
        # ---- 模式①：从命名触发通道解析 (BDF) ----
        if trigger_ch_name not in raw.ch_names:
            available = [ch for ch in raw.ch_names if 'rigger' in ch.lower()
                         or 'tatus' in ch.lower() or 'tim' in ch.lower()]
            hint = ""
            if available:
                hint = (f"\n  可能的触发通道名: {available}"
                        f"\n  请将 trigger_ch_name 设为其中之一。")
            raise ValueError(
                f"触发通道 '{trigger_ch_name}' 不在数据通道列表中。"
                f"{hint}\n"
                f"  所有通道: {raw.ch_names}")

        print(f"[extract] 从触发通道 '{trigger_ch_name}' 提取事件...")
        events = mne.find_events(
            raw,
            stim_channel=trigger_ch_name,
            shortest_event=1,
            mask=255,
            mask_type='and',
            verbose=False)
        print(f"[extract] 共提取 {len(events)} 个事件")

    else:
        # ---- 模式②：从文件标注提取 ----
        if event_map is None:
            event_map = {str(e): e for e in range(1, 255)}

        print(f"[extract] 从文件标注提取事件 "
              f"(标签范围 {min(event_map.values())}~{max(event_map.values())})...")
        events, _ = mne.events_from_annotations(raw, event_id=event_map)
        print(f"[extract] 共提取 {len(events)} 个事件")

    if len(events) == 0:
        raise ValueError("未提取到任何事件！请检查数据文件或 trigger_ch_name 设置。")

    return events


def read_raw_bdf(file_path, trigger_ch_name=None):
    """
    读取 BDF 文件。如果指定了 trigger_ch_name，会将其作为刺激通道加载。

    参数
    ----
    file_path : str
        BDF 文件路径
    trigger_ch_name : str or None
        触发通道名称

    返回
    ----
    raw : mne.io.Raw
    """
    kwargs = dict(preload=True, verbose=False)
    if trigger_ch_name is not None:
        kwargs['stim_channel'] = trigger_ch_name
    return mne.io.read_raw_bdf(file_path, **kwargs)
