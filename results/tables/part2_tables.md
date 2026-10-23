## QPS at recall@10 targets (chosen on tune queries, measured on test)

| dataset   | index        | target   | setting                   |   recall10 |       qps |   speedup_vs_exact |   max_recall10 | note                                    |
|:----------|:-------------|:---------|:--------------------------|-----------:|----------:|-------------------:|---------------:|:----------------------------------------|
| sift      | HNSW         | 0.9      | M=32 efC=100 efSearch=32  |      0.927 | 11567.688 |            355.971 |        nan     | nan                                     |
| sift      | HNSW         | 0.95     | M=32 efC=100 efSearch=48  |      0.957 |  8715.527 |            268.202 |        nan     | nan                                     |
| sift      | IVF-Flat     | 0.9      | nlist=4096 nprobe=32      |      0.932 |  3155.052 |             97.090 |        nan     | nan                                     |
| sift      | IVF-Flat     | 0.95     | nlist=2048 nprobe=32      |      0.961 |  1798.257 |             55.338 |        nan     | nan                                     |
| sift      | IVF-PQ       | 0.9      | not reached               |    nan     |   nan     |            nan     |          0.885 | max 0.885 at nlist=2048 m=64 nprobe=512 |
| sift      | IVF-PQ       | 0.95     | not reached               |    nan     |   nan     |            nan     |          0.885 | max 0.885 at nlist=2048 m=64 nprobe=512 |
| glove     | HNSW         | 0.9      | M=32 efC=100 efSearch=256 |      0.910 |  1428.365 |             42.194 |        nan     | nan                                     |
| glove     | HNSW         | 0.95     | M=32 efC=100 efSearch=768 |      0.961 |   444.600 |             13.133 |        nan     | nan                                     |
| glove     | IVF-Flat     | 0.9      | nlist=1024 nprobe=64      |      0.907 |   567.283 |             16.757 |        nan     | nan                                     |
| glove     | IVF-Flat     | 0.95     | nlist=2048 nprobe=256     |      0.958 |   293.877 |              8.681 |        nan     | nan                                     |
| glove     | IVF-PQ       | 0.9      | not reached               |    nan     |   nan     |            nan     |          0.864 | max 0.864 at nlist=2048 m=50 nprobe=512 |
| glove     | IVF-PQ       | 0.95     | not reached               |    nan     |   nan     |            nan     |          0.864 | max 0.864 at nlist=2048 m=50 nprobe=512 |
| sift      | Exact (Flat) | any      |                           |      1.000 |    32.496 |              1.000 |        nan     | nan                                     |
| glove     | Exact (Flat) | any      |                           |      1.000 |    33.853 |              1.000 |        nan     | nan                                     |

## Build time and size at recall 0.95

| dataset   | index        | setting                                        |   build_s |   train_s |   add_s |   index_mib |   bytes_per_vec |
|:----------|:-------------|:-----------------------------------------------|----------:|----------:|--------:|------------:|----------------:|
| sift      | HNSW         | M=32 efC=100                                   |     65.10 |      0.00 |   65.10 |      747.80 |          784.13 |
| sift      | IVF-Flat     | nlist=2048                                     |     26.34 |     22.05 |    4.29 |      496.93 |          521.07 |
| sift      | IVF-PQ       | nlist=2048 m=64 (max recall; 0.95 not reached) |     88.42 |     76.93 |   11.49 |       69.81 |           73.20 |
| glove     | HNSW         | M=32 efC=100                                   |    103.87 |      0.00 |  103.87 |      758.65 |          672.15 |
| glove     | IVF-Flat     | nlist=2048                                     |     25.38 |     20.64 |    4.74 |      461.30 |          408.71 |
| glove     | IVF-PQ       | nlist=2048 m=50 (max recall; 0.95 not reached) |     74.43 |     63.26 |   11.17 |       66.36 |           58.79 |
| sift      | Exact (Flat) |                                                |      0.28 |    nan    |  nan    |      488.28 |          512.00 |
| glove     | Exact (Flat) |                                                |      0.19 |    nan    |  nan    |      451.47 |          400.00 |
