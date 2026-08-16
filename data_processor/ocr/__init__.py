from paddleocr import PaddleOCR

# Model: PP-OCRv6_medium (largest v6: 34.5M params, multilingual)
# Backend: Paddle Inference
# enable_mkldnn=False: Fix PaddlePaddle 3.3.x PIR + oneDNN incompatibility
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)
result = ocr.predict("data/keyframe/SigLIP/L27_V001/keyframe_14220.webp")

for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
