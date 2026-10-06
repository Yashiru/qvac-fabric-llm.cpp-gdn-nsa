# Run-to-run spread, CUDA, final code (build e34ed08eb), llama-bench -ngl 99 -p 512 -n 128 -r 5, three runs within
# 10 minutes on a desktop session (browser and editors open, load average ~4). Run 1 is bench-cuda.md.
# A first session earlier the same day (binaries built from the then uncommitted working tree, labelled a091d002c) measured the "earlier" column.

| weights | test | run 1 | run 2 | run 3 | earlier |
|---|---|---:|---:|---:|---:|
| F32 | pp512 | 6341 +/- 359 | 6869 +/- 391 | 7062 +/- 277 | 7230 +/- 232 |
| F32 | tg128 | 199.7 +/- 8.9 | 197.6 +/- 10.2 | 202.9 +/- 8.7 | 209.4 +/- 8.0 |
| F16 | pp512 | 9646 +/- 393 | 9196 +/- 502 | 10054 +/- 346 | 10584 +/- 434 |
| F16 | tg128 | 256.3 +/- 12.1 | 250.7 +/- 6.8 | 272.8 +/- 8.3 | 280.2 +/- 6.8 |
| Q8_0 | pp512 | 8992 +/- 615 | 8298 +/- 908 | 9039 +/- 186 | 9444 +/- 176 |
| Q8_0 | tg128 | 315.7 +/- 9.9 | 285.9 +/- 33.2 | 326.7 +/- 13.6 | 337.6 +/- 8.0 |
| Q4_K_M | pp512 | 8218 +/- 645 | 8578 +/- 327 | 9019 +/- 220 | 9160 +/- 168 |
| Q4_K_M | tg128 | 335.1 +/- 12.6 | 318.0 +/- 7.4 | 356.3 +/- 13.8 | 371.2 +/- 11.0 |

Decode cost model fitted per run (t_token = fixed + weight_bytes / bandwidth, 4 weight types):

| run | fixed | bandwidth |
|---|---:|---:|
| run 1 | 2.59 ms | 584 GB/s |
| run 2 | 2.85 ms | 642 GB/s |
| run 3 | 2.40 ms | 565 GB/s |
| earlier | 2.31 ms | 577 GB/s |
