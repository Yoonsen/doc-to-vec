
### mat_338: 338 books, mean |dom(f)| = 706, mean |I| = 265

| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |
|---|---|---|---|---|
| pf_dot | pair | 18.6 µs | 2.1 s (est) | 0.0e+00 |
| align_gallop | pair | 17 µs | 1.9 s (est) | 0.0e+00 |
| align_twopointer | pair | 15.7 µs | 1.8 s (est) | 0.0e+00 |
| align_rank | pair | 116 µs | 13.3 s (est) | 0.0e+00 |
| bitblock | pair | 11.8 µs | 1.3 s (est) | 5.7e-15 |
| ws_dot | pair | 5.73 µs | 0.7 s (est) | 3.3e-16 |
| ws_dot_jsd | pair | 11.8 µs | 1.4 s (est) | 1.7e-15 |
| nb_merge | batched | 0.934 µs | 0.1 s (est) | 0.0e+00 |
| nb_bitblock | batched | 0.102 µs | 0 s (est) | 0.0e+00 |
| nb_workspace | batched | 0.281 µs | 0 s (est) | 0.0e+00 |
| scipy_spgemm | matmul | 0.826 µs | 0.1 s (est) | 1.2e-05 |
| torch_spmm | matmul | 0.158 µs | 0.02 s | 5.3e-07 |
| torch_dense_fp32 | matmul | 0.391 µs | 0.04 s | 4.8e-07 |
| torch_dense_fp16 | matmul | 0.0127 µs | 0 s | 9.8e-04 |

### synthetic_1000: 1,000 books, mean |dom(f)| = 15,527, mean |I| = 4,752

| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |
|---|---|---|---|---|
| pf_dot | pair | 265 µs | 265 s (est) | 0.0e+00 |
| align_gallop | pair | 259 µs | 259 s (est) | 0.0e+00 |
| align_twopointer | pair | 312 µs | 312 s (est) | 0.0e+00 |
| align_rank | pair | 5.9e+03 µs | 98.4 min (est) | 0.0e+00 |
| bitblock | pair | 219 µs | 219 s (est) | 2.1e-13 |
| ws_dot | pair | 207 µs | 207 s (est) | 2.9e-15 |
| ws_dot_jsd | pair | 236 µs | 236 s (est) | 2.3e-13 |
| nb_merge | batched | 7.39 µs | 7.4 s (est) | 0.0e+00 |
| nb_bitblock | batched | 2.62 µs | 2.6 s (est) | 0.0e+00 |
| nb_workspace | batched | 3.42 µs | 3.4 s (est) | 0.0e+00 |
| scipy_spgemm | matmul | 12.6 µs | 12.6 s (est) | 3.5e-04 |
| torch_spmm | matmul | 0.108 µs | 0.11 s | 1.3e-05 |
| torch_dense_fp32 | matmul | 0.085 µs | 0.08 s | 6.6e-05 |
| torch_dense_fp16 | matmul | 0.0189 µs | 0.02 s | 6.3e-03 |

### synthetic_10000: 10,000 books, mean |dom(f)| = 15,466, mean |I| = 4,927

| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |
|---|---|---|---|---|
| pf_dot | pair | 275 µs | 458 min (est) | 0.0e+00 |
| align_gallop | pair | 245 µs | 408 min (est) | 0.0e+00 |
| align_twopointer | pair | 322 µs | 537 min (est) | 0.0e+00 |
| align_rank | pair | 5.88e+03 µs | 6.8 d (est) | 0.0e+00 |
| bitblock | pair | 226 µs | 377 min (est) | 2.2e-13 |
| ws_dot | pair | 154 µs | 257 min (est) | 2.8e-15 |
| ws_dot_jsd | pair | 195 µs | 324 min (est) | 1.2e-13 |
| nb_merge | batched | 6.75 µs | 11.2 min (est) | 0.0e+00 |
| nb_bitblock | batched | 2.25 µs | 226 s (est) | 0.0e+00 |
| nb_workspace | batched | 2.52 µs | 252 s (est) | 0.0e+00 |
| scipy_spgemm | matmul | 9.76 µs | 16.3 min (est) | 3.7e-04 |
| torch_spmm | matmul | 0.0754 µs | 7.54 s | 1.3e-05 |
| torch_dense_fp32 | matmul | 0.0636 µs | 6.36 s | 8.0e-05 |
| torch_dense_fp16 | matmul | 0.0245 µs | 2.45 s | 1.1e-03 |

### shard_8969: 8,969 books, mean |dom(f)| = 8,980, mean |I| = 1,176

| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |
|---|---|---|---|---|
| pf_dot | pair | 87.6 µs | 117 min (est) | 0.0e+00 |
| align_gallop | pair | 76.6 µs | 103 min (est) | 0.0e+00 |
| align_twopointer | pair | 115 µs | 154 min (est) | 0.0e+00 |
| align_rank | pair | 1.02e+03 µs | 22.9 h (est) | 0.0e+00 |
| bitblock | pair | 81.3 µs | 109 min (est) | 6.1e-14 |
| ws_dot | pair | 123 µs | 165 min (est) | 1.6e-15 |
| ws_dot_jsd | pair | 84.9 µs | 114 min (est) | 3.4e-14 |
| nb_merge | batched | 3.61 µs | 290 s (est) | 0.0e+00 |
| nb_bitblock | batched | 1.7 µs | 137 s (est) | 0.0e+00 |
| nb_workspace | batched | 2.23 µs | 179 s (est) | 0.0e+00 |
| scipy_spgemm | matmul | 1.14 µs | 91.9 s (est) | 3.1e-04 |
| torch_spmm | matmul | 0.0505 µs | 4.06 s | 7.4e-06 |
| torch_dense_fp32 | skipped | dense 8969x11812683 needs 394.7 GB | | |
| torch_dense_fp16 | skipped | dense 8969x11812683 needs 197.3 GB | | |

### synthetic_100000: 100,000 books, mean |dom(f)| = 15,608, mean |I| = 4,748

| method | kind | time per pair | all-vs-all (N² dots) | max abs err vs float64 ref |
|---|---|---|---|---|
| pf_dot | pair | 258 µs | 29.9 d (est) | 0.0e+00 |
| align_gallop | pair | 250 µs | 28.9 d (est) | 0.0e+00 |
| align_twopointer | pair | 313 µs | 36.3 d (est) | 0.0e+00 |
| align_rank | pair | 6.76e+03 µs | 783 d (est) | 0.0e+00 |
| bitblock | pair | 217 µs | 25.2 d (est) | 1.9e-13 |
| ws_dot | pair | 166 µs | 19.2 d (est) | 2.9e-15 |
| ws_dot_jsd | pair | 197 µs | 22.8 d (est) | 1.5e-13 |
| nb_merge | batched | 6.51 µs | 18.1 h (est) | 0.0e+00 |
| nb_bitblock | batched | 2.21 µs | 369 min (est) | 0.0e+00 |
| nb_workspace | batched | 2.39 µs | 398 min (est) | 0.0e+00 |
| torch_spmm | matmul | 0.0725 µs | 12.1 min (est) | 1.4e-05 |
| torch_dense_fp32 | skipped | dense 100000x500000 needs 186.3 GB | | |
| torch_dense_fp16 | skipped | dense 100000x500000 needs 93.1 GB | | |
