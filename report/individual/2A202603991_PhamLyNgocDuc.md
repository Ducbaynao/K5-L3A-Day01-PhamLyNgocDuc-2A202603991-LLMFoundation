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

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.885   0.919
2. BoW + MLP (deep learning)     0.937   0.947
3a. Encoder (MiniLM) + kNN       0.890   0.947
3b. Encoder (MiniLM) + LogReg    0.929   0.947
3c. Encoder (MiniLM) + MLP head  0.945   0.947
4a. LLM nhỏ zero-shot            0.000   0.000
4b. LLM lớn zero-shot            0.000   0.000
4c. LLM nhỏ few-shot             0.000   0.000

## 3. Dự đoán và thực tế

F1 của BoW + Logistic Regression (ml_f1). Em dự đoán F1 khoảng 0,85 và precision cao hơn recall. Kết quả trên tập test là F1 = 0,885, precision 0,988 và recall 0,802. Em đoán đúng thứ tự của hai chỉ số nhưng đánh giá F1 thấp hơn thực tế 0,035. Mô hình chỉ gắn nhầm 1 tin ham thành spam ở ngưỡng 0.5, nên precision vẫn rất cao dù mô hình bỏ sót 20 trong 101 tin spam.

Mô hình nằm trên đường Pareto (tradeoff). Em dự đoán ít nhất một mô hình BoW sẽ nằm trên đường Pareto, MiniLM có thể góp mặt nếu cải thiện F1, và cả ba phương án LLM sẽ vượt SLA 50 ms. Kết quả có BoW + LogReg, BoW + MLP và LLM lớn zero-shot trên đường Pareto. MiniLM không góp mặt vì các phương án MiniLM đều đạt F1 = 0,947 trên SUB 40, bằng BoW + MLP nhưng có p95 cao hơn. Dự đoán về SLA của ba phương án LLM khớp kết quả.

Độ bền của LLM trên 50 tin khó (hard_eval). Em dự đoán LLM nhỏ zero-shot đạt khoảng 75% accuracy và gặp khó nhất ở nhóm prompt injection. Thực tế, mô hình đạt 98%, đúng cả 6/6 tin trong nhóm prompt injection và chỉ sai 1/10 tin ở nhóm tưởng spam nhưng không phải. Em đã đánh giá quá cao độ khó của sáu mẫu chèn chỉ dẫn trong bộ này. Lỗi quan sát được lại nằm ở việc phân biệt tin hợp lệ có dấu hiệu giống spam.

## 4. Phân tích 50 case khó

BoW + LogReg giảm F1 nhiều nhất, từ 0,919 trên SUB 40 xuống 0,286 trên 50 tin khó, giảm 0,633. Điểm yếu rõ nhất của mô hình là nhóm spam giọng tự nhiên, với 0/10 tin được nhận diện đúng. Với BoW + MLP, nhóm khó nhất là né từ khóa, với 0/6 tin đúng.
- Case #11 — Khuyến mãi phòng gym viết như tin nhắn của người quen. Tin mở đầu bằng lời chào thân mật rồi mời người nhận phản hồi JOIN. BoW + LogReg gán nhầm thành ham. Một cách giải thích là các từ xuất hiện riêng lẻ giống tin trò chuyện, còn mô hình tuyến tính trên BoW chỉ cộng trọng số từng từ nên khó tận dụng cách chúng kết hợp thành một lời chào mời.
- Case #29 — Viết biến dạng lời chúc trúng thưởng. Tin thay chữ bằng số trong các từ như C0ngr4tulat10ns và Cl4im. BoW + MLP gán nhầm thành ham. Sau khi tách từ, các dạng biến thể được tra như token riêng và có thể nằm ngoài từ điển. Khi tín hiệu đã mất ở bước vector hóa, lớp ẩn cũng không khôi phục được từ gốc.
LLM nhỏ zero-shot đúng 49/50, còn phiên bản có prompt phòng thủ đúng 50/50. Cả hai cùng đúng 6/6 ở nhóm prompt injection. Tin được cải thiện thuộc nhóm tưởng spam nhưng không phải. Bộ thử này giúp tìm điểm yếu theo từng kiểu tin, còn muốn ước tính tỷ lệ lỗi khi triển khai thì cần tập dữ liệu phản ánh lưu lượng thực tế.

## 5. Thử thách cá nhân

Đề #1 theo SEED yêu cầu thay đổi ngưỡng phân loại của BoW + Logistic Regression và chọn ngưỡng cho tình huống ngân hàng muốn tránh chặn nhầm OTP. Em giữ nguyên mô hình đã huấn luyện, tính xác suất spam một lần trên tập test rồi áp dụng ba ngưỡng.
Ngưỡng	Precision	Recall	F1	Tin ham bị chặn nhầm
0,2	0,901	0,901	0,901	10
0,5	0,988	0,802	0,885	1
0,8	1,000	0,535	0,697	0


Nếu ưu tiên tránh chặn nhầm OTP, em chọn 0,8 trong ba ngưỡng đã thử. Ở ngưỡng này, tập test không có tin ham bị chặn nhầm, nhưng mô hình bỏ sót 47/101 tin spam. Ngưỡng 0,2 cho F1 cao nhất và chặn nhầm 10 tin hợp lệ. Trước khi áp dụng cho ngân hàng, em sẽ kiểm tra riêng tỷ lệ chặn nhầm trên một tập validation có nhãn OTP, vì cột cuối của bảng đang đếm tất cả tin ham.

## 6. Phương án cho sản phẩm thật

Em chọn BoW + MLP cho luồng phân loại chính. Trong lần đo trên Colab, mô hình có p95 = 0,26 ms/tin, thấp hơn SLA 50 ms, và đạt F1 = 0,937 trên toàn tập test, 0,947 trên SUB 40. Tốc độ xử lý theo lô đo được là 49.321 tin/giây, so với tải trung bình khoảng 116 tin/giây của 10 triệu SMS/ngày. Hai phương án MiniLM đạt SLA cũng có F1 0,947 trên SUB 40, nhưng độ trễ cao hơn. Để xác nhận SLA khi triển khai, em cần đo p95 đầu cuối ở giờ cao điểm với nhiều yêu cầu đồng thời.

Đánh đổi chính là BoW + MLP chỉ đạt F1 = 0,432 trên 50 tin khó. LLM lớn zero-shot đạt 0,974 trên SUB 40, nhưng p95 khoảng 1.204 ms. Nếu dùng cho toàn bộ lưu lượng trong tháng 30 ngày, chi phí ước tính từ notebook là khoảng 21.750 USD/tháng. Em sẽ dùng BoW + MLP để xử lý ngay các tin có điểm phân loại rõ ràng, chuyển tin lưỡng lự sang MiniLM để xét thêm ngữ nghĩa, và chỉ dùng LLM để rà soát bất đồng bộ các ca còn khó.

## 7. AI log và điều đã học

Trước bài lab, em thường nhìn accuracy trước. Qua mô hình luôn đoán ham, em hiểu rằng accuracy 87,37% vẫn có thể đi cùng với việc bỏ sót toàn bộ spam. Em cũng hiểu vì sao cần xem cả lỗi theo nhóm và độ trễ, vì một mô hình đạt F1 cao trên test thường vẫn có thể gặp khó với tin ngụy trang hoặc không đáp ứng SLA.