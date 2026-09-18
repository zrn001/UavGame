# -*- coding: utf-8 -*-
"""
诊断脚本（一次性工具，不影响主流程）。

用现有 BDF 校准数据，对比【不同导联集合 × 不同分类窗长】的交叉验证准确率，
用数据找出最优配置。不改动任何现有文件。

用法: python diagnose_accuracy.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import mne
mne.set_log_level('ERROR')

# 复用离线训练脚本里现成的函数（不重写）
from offline_TRCA_train import (
    load_raw_file, Preprocessor, segment_epochs,
    leave_one_block_cv, build_fbtrca_model)

# ======================== 配置 ========================
DATA_FILE = r"C:\Users\djm\Documents\DitingBrain\EEG_20260723_18_42_57.bdf"
TRIGGER = "Trigger/Status"
SRATE = 1000
L_FREQ, H_FREQ = 1, 90
N_FOLDS = 4
VISUAL_DELAY = 0.14

# ---- 要对比的【导联集合】 ----
CHANNEL_SETS = {
    "全部31导联(当前)": [
        'PZ','OZ','FC2','O2','C4','FC6','PO6','T8','P8','F8','CP6','P4','F4',
        'CP2','FP2','POZ','F7','P7','CP5','F3','P3','FC1','O1','CZ','FPZ',
        'CP1','FP1','C3','PO5','FC5','T7'],
    "枕区7导联": ['PZ','POZ','PO5','PO6','O1','OZ','O2'],
    "枕顶区11导联": ['PZ','POZ','PO5','PO6','O1','OZ','O2','P3','P4','P7','P8'],
}

# ---- 要对比的【分类窗长】(秒) ----
WINDOWS = [0.5, 0.8, 1.0]

# 滤波器组（与 offline 一致）
WP = [(5, 90), (14, 90), (22, 90), (30, 90)]
WS = [(3, 92), (12, 92), (20, 92), (28, 92)]


def run_one(chan_sel, window):
    """跑一个 (导联集合, 窗长) 组合，返回平均准确率。"""
    tmin = VISUAL_DELAY
    tmax = VISUAL_DELAY + window

    raw, events, ch_names = load_raw_file(DATA_FILE, trigger_ch_name=TRIGGER)
    pre = Preprocessor(
        ch_names=ch_names, fs=SRATE, drop_channels=None, refer=None,
        chan_sel=chan_sel, passband=[L_FREQ, H_FREQ], resample_srate=None)
    data = pre.process(raw)
    X, y = segment_epochs(data, events, fs=SRATE, tmin=tmin, tmax=tmax)

    model = build_fbtrca_model(WP, WS, SRATE, n_components=1)
    results = leave_one_block_cv(X, y, model, n_folds=N_FOLDS, notch_fs=SRATE)
    return results['mean_acc'], results['std_acc']


if __name__ == '__main__':
    print("=" * 70)
    print("   准确率诊断：导联集合 × 窗长  对照实验")
    print(f"   数据: {DATA_FILE}")
    print(f"   交叉验证: {N_FOLDS} 折, 随机水平(6类)=16.7%")
    print("=" * 70)

    table = {}
    for set_name, chans in CHANNEL_SETS.items():
        for win in WINDOWS:
            key = (set_name, win)
            try:
                acc, std = run_one(chans, win)
                table[key] = (acc, std)
                print(f"  [{set_name:16s} | {win}s] "
                      f"准确率 = {acc:.3f} ± {std:.3f}")
            except Exception as e:
                table[key] = (None, None)
                print(f"  [{set_name:16s} | {win}s] 失败: {e}")

    # ---- 汇总表 ----
    print("\n" + "=" * 70)
    print("   汇总（行=导联集合, 列=窗长）")
    print("=" * 70)
    header = "  导联集合          " + "".join(f"  {w}s   " for w in WINDOWS)
    print(header)
    best_key, best_acc = None, -1
    for set_name in CHANNEL_SETS:
        row = f"  {set_name:16s}"
        for win in WINDOWS:
            acc, _ = table[(set_name, win)]
            if acc is None:
                row += "   ----  "
            else:
                row += f"  {acc:.3f} "
                if acc > best_acc:
                    best_acc, best_key = acc, (set_name, win)
        print(row)

    if best_key:
        print("\n" + "=" * 70)
        print(f"  ★ 最优配置: 导联={best_key[0]}, 窗长={best_key[1]}s, "
              f"准确率={best_acc:.3f}")
        print("=" * 70)
