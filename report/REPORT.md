# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/6/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Record này nói về cả bức ảnh. Vì trong hình traffic, có nhiều phương tiện khác nhau, Model phân loại từng phương tiện và dự đoán lớp "cab" đứng hạng 1 với score 0.510915 
4. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Danh sách tên lớp không phải do model tự nghĩ ra, mà đi kèm với taxonomy và cấu hình lớp được sử dụng khi huấn luyện model.
5. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Vì để tránh nhập nhằng, tái lập kết quả sau này, truy vết chính xác nguồn của nhãn.
6. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần quy định rõ tiêu chí lựa chọn lớp đại diện cho toàn bộ bức ảnh.
7. Vì sao không được lấy `score` của model làm ground truth? Vì `score` chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó là chính xác.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên-trái nằm khoảng tại pixel(385, 69). Góc dưới-phải nằm khoảng tại pixel(499, 349). Box rộng khoảng 113.58 pixel. Box cao khoảng 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ngưỡng 0.20 có 17 prediction. Ngưỡng 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Khi hạ ngưỡng khiến **reviewer** phải dành nhiều thời gian QC hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Vẽ bounding box sát nhất có thể quanh phần nhìn thấy được của đối tượng; box phải bao phủ đối tượng nhưng không bao gồm nhiều nền xung quanh hơn mức cần thiết.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Nên đưa vào guideline. Đây là tình huống rất thường gặp, không mơ hồ và có thể chuẩn hóa thành quy tắc chung cho mọi annotator.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446,70],[445,71],[444,71],[443,72],[442,72]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Polygon cho biết: biên thực của vật thể, phần lồi lõm, khoảng thụt vào, hình dáng chân tay của người, phần bị khuyết trong đường viền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? `instance_id` dùng để: định danh duy nhất từng đối tượng được annotate trong chính ảnh đó.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. Vẽ polygon bám sát ranh giới phần nhìn thấy được của vật thể; bao gồm mọi pixel thuộc vật thể và không lấn sang nền hoặc vật thể khác.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? Lò vi sóng bị che đi 1 phần bởi nhiều vật thể khác nhau. Trong trường hợp này dùng guideline. Vì đây là tình huống rất phổ biến trong segmentation, có thể chuẩn hóa thành quy tắc chung, mọi annotator đều có thể áp dụng nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn cho cả ảnh. Ví dụ ảnh traffic: class_id=468, class_name=cab, taxonomy_name=ImageNet-1K.  | Ảnh giao thông có rất nhiều xe buýt, ô tô và người, nên khó quyết định chỉ chọn một lớp đại diện. Model top-1 là cab với score=0.510915, nhưng đây chỉ là dự đoán chứ không phải ground truth.  | Chọn đúng một class theo guideline của taxonomy. | Kiểm tra lớp được chọn có phù hợp quy tắc ưu tiên trong guideline hay không; không dựa vào score của model. |
| Phát hiện vật thể | 1 record = 1 vật thể + 1 bounding box. Ví dụ ảnh kitchen: class_name=person, bbox_xyxy=[385.33,69.24,498.92,348.92]. | Có vật thể bị cắt mép ảnh. Ví dụ bowl có box bbox_xyxy=[0.32,305.02,87.69,371.32], tọa độ sát mép trái ảnh (x≈0). | Vẽ box khít quanh từng vật thể nhìn thấy; gán đúng class. | Kiểm tra box có bao trọn vật thể, không quá rộng, không bỏ sót vật thể, xử lý nhất quán các trường hợp bị cắt mép/che khuất. |
| Instance segmentation | 1 record = 1 vật thể + 1 polygon/mask. Ví dụ instance_id=kitchen-001, class_name=person, polygon_point_count=348. | Biên người đứng giữa bếp bị giao với nhiều vật treo phía sau; ranh giới không phải lúc nào cũng rõ. Record kitchen-001 có polygon rất chi tiết (348 điểm), cho thấy việc xác định biên có thể khó. | Vẽ polygon bám sát phần nhìn thấy của vật thể, không lấn sang nền hoặc vật thể khác. | Kiểm tra đường biên mask có theo đúng mép vật thể không; xử lý nhất quán các vùng biên mờ, vật chạm nhau hoặc bị che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) Chỉ sử dụng dữ liệu trong phạm vi được giao cho mục đích gán nhãn/đánh giá, không sao chép, chia sẻ, tải lên dịch vụ bên ngoài hoặc tiết lộ nội dung cho người không được phép truy cập.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Dừng công việc trên dữ liệu đó ngay lập tức và báo cho Lab Coach/Reviewer hoặc người phụ trách dự án để được hướng dẫn xử lý theo đúng quy trình.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
