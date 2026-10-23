## QPS at recall@10 targets (chosen on tune queries, measured on test)

| dataset   | index        | target   | setting                   |   recall10 |       qps |   speedup_vs_exact |   max_recall10 | note                                    |
|:----------|:-------------|:---------|:--------------------------|-----------:|----------:|-------------------:|---------------:|:----------------------------------------|
| sift      | HNSW         | 0.9      | M=32 efC=100 efSearch=32  |      0.927 | 11381.174 |            350.232 |        nan     | nan                                     |
| sift      | HNSW         | 0.95     | M=32 efC=100 efSearch=48  |      0.958 |  8480.779 |            260.978 |        nan     | nan                                     |
| sift      | IVF-Flat     | 0.9      | nlist=4096 nprobe=32      |      0.932 |  2866.690 |             88.216 |        nan     | nan                                     |
| sift      | IVF-Flat     | 0.95     | nlist=2048 nprobe=32      |      0.961 |  1768.617 |             54.425 |        nan     | nan                                     |
| sift      | IVF-PQ       | 0.9      | not reached               |    nan     |   nan     |            nan     |          0.885 | max 0.885 at nlist=2048 m=64 nprobe=512 |
| sift      | IVF-PQ       | 0.95     | not reached               |    nan     |   nan     |            nan     |          0.885 | max 0.885 at nlist=2048 m=64 nprobe=512 |
| glove     | HNSW         | 0.9      | M=32 efC=100 efSearch=256 |      0.911 |  1425.012 |             42.094 |        nan     | nan                                     |
| glove     | HNSW         | 0.95     | M=32 efC=100 efSearch=768 |      0.962 |   440.852 |             13.023 |        nan     | nan                                     |
| glove     | IVF-Flat     | 0.9      | nlist=2048 nprobe=128     |      0.920 |   537.423 |             15.875 |        nan     | nan                                     |
| glove     | IVF-Flat     | 0.95     | nlist=2048 nprobe=256     |      0.958 |   301.033 |              8.892 |        nan     | nan                                     |
| glove     | IVF-PQ       | 0.9      | not reached               |    nan     |   nan     |            nan     |          0.864 | max 0.864 at nlist=2048 m=50 nprobe=512 |
| glove     | IVF-PQ       | 0.95     | not reached               |    nan     |   nan     |            nan     |          0.864 | max 0.864 at nlist=2048 m=50 nprobe=512 |
| sift      | Exact (Flat) | any      |                           |      1.000 |    32.496 |              1.000 |        nan     | nan                                     |
| glove     | Exact (Flat) | any      |                           |      1.000 |    33.853 |              1.000 |        nan     | nan                                     |

## Build time and size at recall 0.95

| dataset   | index        | setting                                        |   build_s |   train_s |   add_s |   index_mib |   bytes_per_vec |
|:----------|:-------------|:-----------------------------------------------|----------:|----------:|--------:|------------:|----------------:|
| sift      | HNSW         | M=32 efC=100                                   |     65.51 |      0.00 |   65.51 |      747.80 |          784.13 |
| sift      | IVF-Flat     | nlist=2048                                     |     26.41 |     22.11 |    4.31 |      496.93 |          521.07 |
| sift      | IVF-PQ       | nlist=2048 m=64 (max recall; 0.95 not reached) |     93.48 |     82.11 |   11.37 |       69.81 |           73.20 |
| glove     | HNSW         | M=32 efC=100                                   |    104.49 |      0.00 |  104.49 |      758.65 |          672.15 |
| glove     | IVF-Flat     | nlist=2048                                     |     25.32 |     20.59 |    4.73 |      461.30 |          408.71 |
| glove     | IVF-PQ       | nlist=2048 m=50 (max recall; 0.95 not reached) |     72.86 |     61.85 |   11.01 |       66.36 |           58.79 |
| sift      | Exact (Flat) |                                                |      0.28 |    nan    |  nan    |      488.28 |          512.00 |
| glove     | Exact (Flat) |                                                |      0.19 |    nan    |  nan    |      451.47 |          400.00 |
