# NCNN Face Recognition – Formal C++/Python Project


## Main program

### `face_main`

include two models：

- `--mode infer`：inference mode。
- `--mode enroll --name NAME`：register new people。


### `face_mini`

mini version, inference only, can be used in command.

### `face_debug`

same with `face_main`, but ouput extra infomation：

```text
image_load
detector_preprocess
detector_inference
detector_postprocess
alignment
recognizer_inference
database_match
database_io
total
```



## Raspberry Pi Compile

require `OpenCV` and `NCNN`。

```bash
mkdir -p build
cd build
cmake -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn ..
cmake --build . -j4
```


## infer

```bash
./build/face_main \
  --mode infer \
  --image test.jpg \
  --det-param models/scrfd_320_int8.param \
  --det-bin models/scrfd_320_int8.bin \
  --rec-param models/mobilefacenet_int8.param \
  --rec-bin models/mobilefacenet_int8.bin \
  --det-size 320 \
  --db db/mobilefacenet_int8.db \
  --threshold 0.45 \
  --threads 4
```

stdout：

```json
{"status":"ok","mode":"infer","image":{"width":1920,"height":1080},"detector_input_size":320,"face_count":2,"faces":[{"username":"Alice","accuracy":0.713421,"metric":"cosine_similarity","known":true,"detection_score":0.941203,"bbox":[...],"landmarks":[...]},{"username":"UNKNOWN","accuracy":0.287315,"metric":"cosine_similarity","known":false,"detection_score":0.911204,"bbox":[...],"landmarks":[...]}]}
```

## enrollment

```bash
./build/face_main \
  --mode enroll \
  --name "Alice" \
  --image alice_new.jpg \
  --det-param models/scrfd_320_fp32.param \
  --det-bin models/scrfd_320_fp32.bin \
  --rec-param models/mobilefacenet_fp32.param \
  --rec-bin models/mobilefacenet_fp32.bin \
  --det-size 320 \
  --db db/mobilefacenet_fp32.db
```

If Alice is in db：

```json
{"status":"ok","mode":"enroll","action":"already_known",...}
```

If Alice is UNKNOWN ：

```json
{"status":"ok","mode":"enroll","action":"enrolled",...}
```

## Database

readable design：

```text
# FACE_DB_V2 embedding_dim=512 templates=2
"Alice" 0.0123 ... 512 values
"Alice" -0.0312 ... 512 values
```

```text
db/mobilefacenet_fp32.db
db/mobilefacenet_fp16.db
db/mobilefacenet_int8.db
```
