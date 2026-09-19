"""Compare the periods detected by the FFT and DCT branches on the real datasets.

Run:  python check_periods.py
Prints, for each dataset, the FFT vs DCT Top-K periods (in steps and in
physical time), how much the two sets overlap, and the union size. This is the
empirical check behind the "dual-domain complementarity" claim: if the two sets
largely coincide, DCT is redundant and the claim does not hold.
"""
import argparse

import numpy as np
import torch

from data_provider.data_factory import data_provider
from models.TimesNetFull import Model, dual_domain_period_detection
from train_solar_full import infer_num_vars

DATASETS = [
    # name,        file,                       sampling min, steps per day
    ("Solar-DKA", "DKA Solar Center dataset.csv", 5, 288),
    ("Solar-Site1", "site1.xlsx", 15, 96),
]


def build_args(data, path, seq_len):
    n_var = infer_num_vars("./data_provider/", path)
    return argparse.Namespace(
        task_name="long_term_forecast", data=data, root_path="./data_provider/",
        data_path=path, features="M", target="Active_Power", freq="t",
        embed="timeF", scale=True, seasonal_patterns="Monthly",
        seq_len=seq_len, label_len=48, pred_len=96, batch_size=32,
        num_workers=0, enc_in=n_var, dec_in=n_var, c_out=n_var,
        d_model=64, d_ff=64,
        e_layers=2, d_layers=1, top_k=5, num_kernels=6, dropout=0.1,
        use_stl=True, stl_period=24, stl_impl="torch", use_autocorr=True,
        use_dct=True,
    )


def main():
    for name, path, sample_min, spd in DATASETS:
        # T must satisfy T >= 2p for the STL path; use 2 daily cycles.
        seq_len = 2 * spd
        args = build_args(name.lower().replace("-", "_"), path, seq_len)
        ds, loader = data_provider(args, "train")
        x, _, _, _ = next(iter(loader))          # x: [B, T, C]

        fft_p, fft_w, dct_p, dct_w = dual_domain_period_detection(x, k=5, use_dct=True)

        def phys(p):
            return p * sample_min / 60.0          # hours

        print("=" * 74)
        print("%s   (采样 %d min, seq_len=%d, 1天=%d步)" % (name, sample_min, seq_len, spd))
        print("-" * 74)
        print("  %-4s %-16s %-16s" % ("rank", "FFT period", "DCT period"))
        for i in range(len(fft_p)):
            print("  %-4d %-16s %-16s" % (
                i + 1,
                "%4d 步 (%.1f h)" % (fft_p[i], phys(fft_p[i])),
                "%4d 步 (%.1f h)" % (dct_p[i], phys(dct_p[i])),
            ))
        fs, ds_ = set(int(v) for v in fft_p), set(int(v) for v in dct_p)
        inter, union = fs & ds_, fs | ds_
        print("-" * 74)
        print("  交集 %s   (重叠 %d/%d)" % (sorted(inter), len(inter), len(union)))
        print("  并集 %s" % sorted(union))
        print("  两路集合是否相同: %s" % ("是 —— DCT 冗余" if fs == ds_ else "否"))

        # Does either branch miss the daily cycle?
        print("  日周期(%d步)是否被检出: FFT=%s  DCT=%s" % (
            spd, spd in fs, spd in ds_))

        # Embedded representation: what the times blocks actually see.
        model = Model(args)
        with torch.no_grad():
            means = x.mean(1, keepdim=True).detach()
            xn = (x - means) / torch.sqrt(torch.var(x, 1, keepdim=True, unbiased=False) + 1e-5)
            trend, seas, resid = model.stl_decomposer(xn)
            mark = torch.zeros(x.shape[0], x.shape[1], 5)
            emb = (model.trend_embedding(trend, mark)
                   + model.seasonal_embedding(seas, mark)
                   + model.residual_embedding(resid, mark))
            emb = model.predict_linear(emb.permute(0, 2, 1)).permute(0, 2, 1)
            f2, _, d2, _ = dual_domain_period_detection(emb, k=5, use_dct=True)
        print("-" * 74)
        print("  [嵌入表示上重新检测] FFT=%s" % list(f2))
        print("                        DCT=%s" % list(d2))
        print("    重叠 %d/5, 日周期检出: FFT=%s DCT=%s" % (
            len(set(f2) & set(d2)), spd in set(f2), spd in set(d2)))
        print()


if __name__ == "__main__":
    main()
