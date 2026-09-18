import re
from typing import Dict, List

FEW_SHOT_EXAMPLE = """[GÓC NHÌN & CỠ CẢNH]
- Cỡ cảnh: Cận cảnh chuyển sang trung cảnh.
- Góc nhìn: Góc quay từ trên xuống chếch khoảng 45 độ, tập trung vào mặt bàn bếp.
- Chuyển động máy quay: Máy quay tĩnh ở đầu shot, sau đó lia nhẹ sang phải theo hướng tay người đầu bếp.

[CHỦ THỂ & HÀNH ĐỘNG]
- Một người đầu bếp mặc áo trắng, đeo tạp dề màu đen, chỉ nhìn thấy hai cánh tay và nửa thân người.
- Hai tay cầm đôi đũa dài bằng gỗ liên tục đảo đều nguyên liệu trong chảo. Tay trái giữ nhẹ cán chảo màu đen, tay phải đảo thức ăn từ mép vào giữa.

[VẬT THỂ & ĐẶC ĐIỂM TRỰC QUAN]
- Một chảo chống dính hình tròn màu đỏ đặt trên bếp từ mặt kính đen.
- Bên trong chảo chứa các miếng thực phẩm màu trắng cắt hình que dài, khi đảo dần nở to phồng lên có màu vàng nhạt.
- Bên cạnh bếp có một chiếc bát sứ màu trắng chứa chất lỏng màu vàng nhạt và một đĩa thủy tinh đựng rau thơm màu xanh lá.

[BỐI CẢNH & KHÔNG GIAN]
- Không gian bếp trong nhà, mặt bàn bếp ốp đá hoa cương màu xám có vân trắng.
- Hậu cảnh mờ ở phía trên là tường bếp ốp gạch men màu trắng.

[CHỮ, LOGO & MÀN HÌNH]
- Logo "VTV3" màu đỏ trắng ở góc trên cùng bên phải.
- Dòng chữ phụ đề màu vàng ở góc dưới màn hình: "Bí quyết chiên phồng tôm giòn xốp".

[DIỄN BIẾN THEO THỜI GIAN]
- Đầu shot: Người đầu bếp thả nguyên liệu màu trắng dạng que dẹt vào chảo dầu đang sôi sủi bọt.
- Giữa shot: Nguyên liệu bắt đầu nở phồng to gấp đôi, dính nhẹ vào nhau, người đầu bếp dùng đũa tách rời.
- Cuối shot: Các miếng nguyên liệu đã phồng đều chuyển màu vàng, chuẩn bị được vớt ra đĩa trắng.

[TỔNG THỂ CẢNH QUAY]
Cảnh quay hướng dẫn nấu ăn ghi lại quá trình chiên một loại thực phẩm màu trắng trong chảo chống dính màu đỏ. Người đầu bếp sử dụng đũa gỗ để đảo và tách các sợi nguyên liệu đang nở phồng trong dầu sôi trên mặt bếp từ."""


def build_shot_prompt(
    t_start: float,
    t_end: float,
    ocr_texts: List[str],
    audio_transcript: str
) -> str:
    ocr_str = ", ".join([f'"{t}"' for t in ocr_texts]) if ocr_texts else "Không phát hiện chữ trên màn hình"
    audio_str = audio_transcript.strip() if audio_transcript.strip() else "Không có lời thoại âm thanh"

    prompt = f"""Bạn là chuyên gia phân tích và chú thích video cho hệ thống Video Retrieval (truy vấn video bằng ngôn ngữ tự nhiên).
Hãy phân tích chuỗi các khung hình đại diện cho MỘT CÚ QUAY (SHOT), kết hợp với OCR và lời thoại âm thanh được cung cấp bên dưới.

Mục tiêu quan trọng nhất: Tạo ra mô tả chứa các thông tin trực quan CỤ THỂ, ĐẶC TRƯNG (DISTINCTIVE) và giàu từ khóa tìm kiếm thực tế.

[THÔNG TIN BỔ TRỢ CỦA SHOT]
- Thời lượng shot: {t_start:.2f}s - {t_end:.2f}s
- Chữ xuất hiện trên màn hình (OCR): {ocr_str}
- Lời thoại âm thanh (Whisper): "{audio_str}"

[NGUYÊN TẮC CỐT LÕI]
1. CHỈ MÔ TẢ NHỮNG GÌ QUAN SÁT ĐƯỢC TRỰC QUAN:
- Tuyệt đối không suy đoán danh tính, địa danh, nghề nghiệp, cảm xúc ("vui vẻ", "căng thẳng") hoặc tính từ chủ quan ("chuyên nghiệp", "đẹp mắt", "ngon miệng").
- Không biến thông tin từ lời thoại/OCR thành đặc điểm thị giác nếu hình ảnh không hiển thị (ví dụ: audio nhắc đến một địa danh hoặc chủ đề nhưng hình ảnh chỉ là trường quay/phòng học thì chỉ mô tả trường quay/phòng học).
- Nếu không chắc chắn tên chính xác của đồ vật/món ăn, hãy mô tả bằng hình dáng, màu sắc và chất liệu thô (ví dụ: "vật liệu màu trắng dạng sợi dính nở to", "khối chất lỏng màu vàng").

2. BÁM SÁT CÁC ĐẶC TRƯNG TÌM KIẾM ĐẶC THÙ:
- Góc quay & Cỡ cảnh: Xác định rõ góc nhìn (ngang tầm mắt, từ trên xuống / top-down, chéo từ dưới lên, từ sau lưng, góc nhìn thứ nhất POV) và cỡ cảnh (toàn/trung/cận/đặc tả). Chuyển động máy quay (máy quay tĩnh, lia trái/phải, phóng to/zoom in, bám theo đối tượng).
- Tư thế & Chi tiết động tác: Mô tả chi tiết tay, chân, hướng nhìn, động tác vi mô (co một chân, đứng trên mấy cọc/trụ, ngậm thanh trụ, nằm dài trên yên xe, hai tay cầm vật gì, cử chỉ tay khi nói).
- Vật thể, màu sắc & trạng thái thô: Mô tả rõ màu sắc đồ chứa/dụng cụ (bát trắng, chảo đỏ, nồi thủy tinh, khay gỗ), chất liệu và trạng thái biến đổi vật lý (dạng que dính vào nhau, nở phồng to khi chiên/nấu, cắt đôi nhưng không tách rời, khứa caro vuông góc, chất lỏng sủi bọt).
- Sơ đồ, Bài giảng & Đề thi (nếu có): Bố cục slide (người đứng bên trái/phải màn hình), hình học (đường xiên, hình phẳng, nét đứt, điểm vuông góc), cấu trúc sơ đồ (dải tròn bắc cầu), câu hỏi trắc nghiệm (câu số mấy, các đáp án A/B/C/D).

3. ĐỊNH DẠNG ĐẦU RA:
- Không lặp lại thông tin nếu không có thay đổi xuyên suốt shot.
- KHÔNG sử dụng định dạng JSON. Chỉ sử dụng tiếng Việt.
- Bắt buộc trả về đúng 7 tiêu đề trong dấu ngoặc vuông `[...]` như cấu trúc mẫu dưới đây.

[VÍ DỤ MẪU ĐẦU RA CHUẨN]
---
{FEW_SHOT_EXAMPLE}
---

Bây giờ, hãy phân tích chuỗi khung hình được cung cấp và xuất ra kết quả theo đúng 7 tiêu đề trên:"""
    return prompt


def parse_aspects(caption_text: str) -> Dict[str, str]:
    """
    Parses aspect sections from the model output.
    Returns a dictionary mapping normalized aspect keys to content strings.
    """
    # Allow any trailing non-newline characters after ] (e.g. ]--> or ]: )
    pattern = r"\[(.*?)\][^\n]*\n(.*?)(?=\n\s*\[|$)"
    matches = re.findall(pattern, caption_text.strip(), re.DOTALL)
    
    aspect_map = {
        "goc_nhin_co_canh": "",
        "chu_the_hanh_dong": "",
        "vat_the_dac_diem": "",
        "boi_canh_khong_gian": "",
        "chu_logo_man_hinh": "",
        "dien_bien_thoi_gian": "",
        "tong_the_canh_quay": "",
    }
    
    for raw_title, content in matches:
        norm = raw_title.strip().lower()
        
        if any(w in norm for w in ["goc", "góc", "cỡ cảnh", "co canh"]):
            aspect_map["goc_nhin_co_canh"] = content.strip()
        elif any(w in norm for w in ["chu the", "chủ thể", "hanh dong", "hành động"]):
            aspect_map["chu_the_hanh_dong"] = content.strip()
        elif any(w in norm for w in ["vat the", "vật thể", "dac diem", "đặc điểm"]):
            aspect_map["vat_the_dac_diem"] = content.strip()
        elif any(w in norm for w in ["boi canh", "bối cảnh", "khong gian", "không gian"]):
            aspect_map["boi_canh_khong_gian"] = content.strip()
        elif any(w in norm for w in ["logo", "man hinh", "màn hình", "chu,"]):
            aspect_map["chu_logo_man_hinh"] = content.strip()
        elif any(w in norm for w in ["dien bien", "diễn biến", "thoi gian", "thời gian"]):
            aspect_map["dien_bien_thoi_gian"] = content.strip()
        elif any(w in norm for w in ["tong the", "tổng thể", "canh quay"]):
            aspect_map["tong_the_canh_quay"] = content.strip()
        else:
            clean_raw = re.sub(r'[^a-zA-Z0-9_]+', '_', norm).strip('_')
            aspect_map[clean_raw] = content.strip()
            
    return aspect_map

