# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603991*
- **Họ và tên:*Phạm Lý Ngọc Đức*
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

## 3. Dự đoán và thực tế

F1 của BoW + Logistic Regression (ml_f1). Em dự đoán F1 khoảng 0,85 và precision cao hơn recall. Kết quả trên tập test là F1 = 0,885, precision 0,988 và recall 0,802. Em đoán đúng thứ tự của hai chỉ số nhưng đánh giá F1 thấp hơn thực tế 0,035. Mô hình chỉ gắn nhầm 1 tin ham thành spam ở ngưỡng 0.5, nên precision vẫn rất cao dù mô hình bỏ sót 20 trong 101 tin spam.

Mô hình nằm trên đường Pareto (tradeoff). Em dự đoán ít nhất một mô hình BoW sẽ nằm trên đường Pareto, MiniLM có thể góp mặt nếu cải thiện F1, và cả ba phương án LLM sẽ vượt SLA 50 ms. Kết quả có BoW + LogReg, BoW + MLP và LLM lớn zero-shot trên đường Pareto. MiniLM không góp mặt vì các phương án MiniLM đều đạt F1 = 0,947 trên SUB 40, bằng BoW + MLP nhưng có p95 cao hơn. Dự đoán về SLA của ba phương án LLM khớp kết quả.

Độ bền của LLM trên 50 tin khó (hard_eval). Em dự đoán LLM nhỏ zero-shot đạt khoảng 75% accuracy và gặp khó nhất ở nhóm prompt injection. Thực tế, mô hình đạt 98%, đúng cả 6/6 tin trong nhóm prompt injection và chỉ sai 1/10 tin ở nhóm tưởng spam nhưng không phải. Em đã đánh giá quá cao độ khó của sáu mẫu chèn chỉ dẫn trong bộ này. Lỗi quan sát được lại nằm ở việc phân biệt tin hợp lệ có dấu hiệu giống spam.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
