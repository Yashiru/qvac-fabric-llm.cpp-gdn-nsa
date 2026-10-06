| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |       6 |           pp512 |        356.56 ± 8.39 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |       6 |           tg128 |         14.02 ± 0.58 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      12 |           pp512 |        389.64 ± 4.72 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      12 |           tg128 |         14.34 ± 0.90 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      20 |           pp512 |        563.04 ± 8.80 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CPU        |      20 |           tg128 |         13.20 ± 0.33 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |       6 |           pp512 |        394.46 ± 8.63 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |       6 |           tg128 |         28.91 ± 1.13 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      12 |           pp512 |       442.62 ± 14.73 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      12 |           tg128 |         25.61 ± 0.41 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      20 |           pp512 |       494.11 ± 15.16 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CPU        |      20 |           tg128 |         22.31 ± 1.03 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |       6 |           pp512 |       422.79 ± 17.17 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |       6 |           tg128 |         47.63 ± 2.02 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      12 |           pp512 |        381.09 ± 9.96 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      12 |           tg128 |         39.17 ± 1.22 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      20 |           pp512 |        455.49 ± 9.63 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CPU        |      20 |           tg128 |         34.02 ± 0.77 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |       6 |           pp512 |        257.71 ± 0.57 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |       6 |           tg128 |         67.27 ± 4.28 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      12 |           pp512 |       486.05 ± 19.69 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      12 |           tg128 |         51.94 ± 1.15 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      20 |           pp512 |       578.66 ± 17.21 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CPU        |      20 |           tg128 |         41.53 ± 2.74 |

build: e34ed08eb (9253)
