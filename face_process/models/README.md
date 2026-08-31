# Models

Place the NCNN `.param` and `.bin` files here (or pass absolute paths from Python).

Verified model interfaces from the supplied `.param` files:

## SCRFD 500M KPS
- input: `in0`
- stride 8: score=`out0`, bbox=`out3`, kps=`out6`
- stride 16: score=`out1`, bbox=`out4`, kps=`out7`
- stride 32: score=`out2`, bbox=`out5`, kps=`out8`
- 2 anchors per feature-map location
- supported detector sizes in this project: 160, 320, 480, 640

## MobileFaceNet
- input: `in0`
- output: `out0`
- embedding dimension: 512
