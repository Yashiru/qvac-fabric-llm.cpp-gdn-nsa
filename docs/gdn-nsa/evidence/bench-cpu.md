| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |       6 |           pp512 |       324.76 ± 13.10 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |       6 |           tg128 |         15.87 ± 0.76 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      12 |           pp512 |        495.27 ± 3.04 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      12 |           tg128 |         15.30 ± 0.03 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      20 |           pp512 |        539.47 ± 5.05 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      20 |           tg128 |         14.03 ± 0.09 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |       6 |           pp512 |        270.52 ± 1.03 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |       6 |           tg128 |         31.03 ± 0.19 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      12 |           pp512 |        365.65 ± 3.84 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      12 |           tg128 |         26.54 ± 0.04 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      20 |           pp512 |        465.70 ± 1.26 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      20 |           tg128 |         23.60 ± 0.03 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |       6 |           pp512 |       448.90 ± 20.82 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |       6 |           tg128 |         49.62 ± 1.38 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      12 |           pp512 |        398.93 ± 1.65 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      12 |           tg128 |         42.27 ± 0.26 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      20 |           pp512 |        479.31 ± 4.47 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      20 |           tg128 |         35.94 ± 0.11 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |       6 |           pp512 |        473.93 ± 2.41 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |       6 |           tg128 |         65.79 ± 1.47 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      12 |           pp512 |        517.51 ± 5.56 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      12 |           tg128 |         57.23 ± 1.68 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      20 |           pp512 |        638.39 ± 5.05 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      20 |           tg128 |         46.01 ± 0.57 |

build: a091d002c (9244)
