# Model weights

| file | in git? | size (bytes) | SHA256 | what |
|---|---|---|---|---|
| `models/detector/cell_yolo26s_v2_best.pt` | yes | 20300421 | `e85d0419bd6dc984772039196546b10996ba784b88ca43ec2be47056380c1451` | YOLO26s @640, cell detector (Ultralytics 8.4.183) |
| `models/seg/v2_1/best.pt` | yes | 548101 | `ab16420ea57123d87e0380ce5f2971410a03d25d5d25f29be19c1fc1e062647b` | TinyUNet v2_1 (default segmenter, 3-ch, 224 px) |
| `models/seg/v2_1/config.json` | yes | 25 | `ea2c3028638a8ef76629accd9f539f107a8c707d7a50176accab318a733d965f` | v2_1 input config |
| `models/seg/v2/best.pt` | yes | 548677 | `01062258bf4e4bb6ed58e0d8d9783426d92204c6524e3ffac51fc36ace56a4f2` | TinyUNet v2 (4-ch box variant; parity tests) |
| `models/seg/v2/config.json` | yes | 25 | `66454e42b88aebc21c607c1e58f593ff04afb42962ddebdd2943b00a0dfa4eea` | v2 input config |
| `models/detector_b/yolo26l_1280_best.pt` | no: Release v0.1-models (scripts/fetch_release_weights.py) | 105729697 | `703f3e716a5116b98d581497eed6bab62b49181577aeb2062f209cbb230fb72b` | YOLO26l @1280, cross-check detector (consensus_flag) |

Every run records these SHA256s in `run_manifest.json`.
