| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |           pp512 |     6341.33 ± 359.04 |
| gdn-nsa 350M all F32           |   1.33 GiB |   358.35 M | CUDA       |  99 |           tg128 |        199.72 ± 8.90 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CUDA       |  99 |           pp512 |     9646.34 ± 392.99 |
| gdn-nsa 350M F16               | 683.89 MiB |   358.35 M | CUDA       |  99 |           tg128 |       256.25 ± 12.13 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CUDA       |  99 |           pp512 |     8992.27 ± 614.88 |
| gdn-nsa 350M Q8_0              | 363.67 MiB |   358.35 M | CUDA       |  99 |           tg128 |        315.72 ± 9.87 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CUDA       |  99 |           pp512 |     8217.63 ± 645.35 |
| gdn-nsa 350M Q4_K - Medium     | 214.29 MiB |   358.35 M | CUDA       |  99 |           tg128 |       335.12 ± 12.58 |

build: e34ed08eb (9253)
