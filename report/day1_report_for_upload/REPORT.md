# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:*6/11/2026*

**Runtime Colab:** GPU(T4)

**Python / PyTorch / Ultralytics: python 3.13.16/PyTourch 2.11.0+cu130/ Ultralytics 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

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

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Tác vụ thực hiện là Image Classification (task: image_classification). Mô hình đưa ra một phân phối xác suất trên toàn thể khung hình mà hoàn toàn không có tọa độ bounding box hay phạm vi pixel cụ thể cho từng phương tiện riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp đi kèm theo taxonomy của checkpoint (ở đây là bộ taxonomy cố định ImageNet-1K gồm 1.000 lớp có sẵn trong checkpoint yolo11n-cls.pt). Mô hình chỉ tính toán phân bổ xác suất trên danh sách này chứ không thể tự sinh thêm tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Để xác định rõ không gian lớp (label space) và tránh nhập nhằng ngữ nghĩa. Cùng một khái niệm (như xe cộ) nhưng taxonomy ImageNet-1K chia chi tiết thành cab, minibus, police_van, còn taxonomy COCO chỉ có nhãn bao quát là car hoặc bus. class_id đảm bảo máy xử lý chính xác không phụ thuộc ngôn ngữ, còn taxonomy_name giúp định vị chuẩn gốc của bộ dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline phải đưa ra tiêu chí ưu tiên chủ thể chính rõ ràng:Ưu tiên vật thể nằm ở vị trí trung tâm hoặc chiếm tỷ lệ diện tích lớn nhất ở tiền cảnh (foreground)hoặc quy định thứ tự ưu tiên đối tượng theo mục tiêu ứng dụng (ví dụ: bối cảnh giao thông thì ưu tiên phương tiện nổi bật hay không gian toàn cảnh).
6. Vì sao không được lấy `score` của model làm ground truth?
   Score (cụ thể là 0.510915) chỉ là độ tin cậy mang tính xác suất của thuật toán, dễ bị ảnh hưởng bởi góc chụp, ánh sáng hay nhiễu. Ngay trên biểu đồ classification_top5.png cũng có chú thích: "độ tin cậy của mô hình (không phải chất lượng ground truth)". Ground truth phải là sự thật khách quan do con người xác thực theo guideline và mang giá trị nhị phân (có hay không có thuộc tính đó), không phải là điểm số xác suất.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=kitchen ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?Hộp bao của người đầu bếp bắt đầu từ góc trên-trái tại tọa độ $(x_1 = 385.33, y_1 = 69.24)$ pixel (vùng đầu/tóc) và kết thúc tại góc dưới-phải ở tọa độ $(x_2 = 498.92, y_2 = 348.92)$ pixel (ngang đầu gối/gấu tạp dề).Chiều rộng của hộp bao là $113.58$ pixel ($x_2 - x_1$) và chiều cao là $279.68$ pixel ($y_2 - y_1$).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ngưỡng 0.20: Có 17 predictions bao gồm các nhãn ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']. Ngưỡng 0.60: Có 6 predictions gồm các đối tượng có độ tin cậy cao nhất ['person', 'bowl', 'bowl', 'oven', 'oven', 'person'].
1. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Khi hạ ngưỡng từ 0.60 xuống 0.20 (từ 6 hộp vọt lên 17 hộp), mô hình bắt đầu giữ lại nhiều dự đoán có độ tin cậy thấp và nhiễu (như đoán thêm spoon, potted plant, dining table, bottle). Người reviewer phải tốn nhiều thời gian và công sức để rà soát từng hộp, xóa thủ công các hộp dự đoán sai (False Positives) hoặc điều chỉnh các hộp bị vẽ lem, trùng lặp chồng đè lên nhau.
2. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Quy tắc: "Hộp bao bounding box phải vẽ sát khít viền ngoài cùng của toàn bộ phần nhìn thấy được của vật thể. tuyệt đối không cắt lẹm vào bộ phận của đối tượng và không chừa khoảng trống viền ra nền xung quanh quá 2–3 pixel."
3. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Quan sát: Hộp bao person có score = 0.610931 tại tọa độ $[0.12, 263.18, 61.55, 310.78]$ thực chất chỉ là cánh tay trái của một người đứng ngoài khung hình đưa vào, bị cắt cụt hoàn toàn phần thân trên và mặt. Mô hình đã nhận diện nhầm cánh tay này thành một "person" độc lập. Cách xử lý: Trường hợp này cần được quy định rõ ràng trong Guideline: "Không gán nhãn person nếu chỉ nhìn thấy một phần chi (tay/chân) bị cắt mép mà không thấy thân mình hoặc khuôn mặt (diện tích thấy được $< 20\%$)". Việc đưa vào guideline là bắt buộc vì hiện tượng người/vật thể bị cắt ngang mép khung hình xảy ra rất thường xuyên trong thực tế gán nhãn, nếu ca nào cũng escalate sẽ làm tê liệt quy trình của Reviewer/Coach.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Bounding box chỉ là hình chữ nhật bao quanh nên chứa nhiều phần phông nền thừa (ví dụ: khoảng trống giữa hai chân hoặc hai bên hông người đầu bếp). Polygon cung cấp đường bao chính xác tới cấp độ từng pixel (pixel-level mask): uốn lượn ôm sát theo đường cong của bờ vai, quai đeo quần, eo và đường cắt của chiếc tạp dề, loại bỏ hoàn toàn phần phông nền xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? instance_id (ví dụ: kitchen-001, kitchen-002) dùng để đánh số và tách biệt từng thực thể riêng biệt trong cùng một khung hình tĩnh. Nó không phải class ID vì nhiều instance có thể chung một nhãn lớp (trong ảnh kitchen có kitchen-002, kitchen-003, kitchen-006, kitchen-010 đều mang class_id = 45 là bowl). Nó không phải tracking ID vì nó chỉ có giá trị nội bộ trong một bức ảnh tĩnh, không bám theo vết chuyển động của đối tượng xuyên suốt các khung hình video theo thời gian.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. Quy tắc: "Đường đa giác polygon phải ôm khít đường viền thực tế của đối tượng; tại các khúc cua bo tròn hoặc chi tiết phức tạp phải tăng mật độ điểm chấm (tối thiểu 30–50 điểm đối với vật thể lớn/vừa) để viền mượt mà; không kéo điểm đè lấn sang đồ vật bên cạnh hoặc để lọt chi tiết nền vào bên trong mask."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? Trường hợp biên mờ (Blurry boundary): Khu vực chùm lá khô / bó thảo mộc treo ở góc trên cùng bên trái (nơi mô hình gán nhãn potted plant 0.63 / kitchen-004). Ranh giới giữa các nhánh lá vụn với mảng tường gỗ tối phía sau và chiếc chảo kim loại bên cạnh bị nhòe mờ do thiếu sáng và độ sâu trường ảnh, khiến viền ngoài của vật thể không hề có đường ranh sắc nét.Trường hợp hai vật chạm nhau và bị che (Touch & Occlusion): Chiếc bàn gỗ ở tiền cảnh (dining table 0.58 / kitchen-005) bị các đồ vật khác (bát nhào bột bowl 0.74, vệt bột mì, đĩa) đè chồng lên trên. Mask màu tím của dining table đã nuốt trọn và phủ đè lên toàn bộ các vật thể đặt bên trên nó. Cách xử lý: Dùng Guideline: Với trường hợp biên mờ (chùm lá/thảo mộc): Quy định annotator vẽ đường bao đa giác gộp chung (convex hull) bao quát hình dáng tổng thể của chùm lá, không cần cố chấm chi tiết từng mẩu lá li ti bị nhòe vào nền tối. Với trường hợp đồ vật đặt trên bàn: Quy định mask của bàn phải khoét rỗng (trừ vùng) các đồ vật nằm đè lên trên nếu đồ vật đó đã được gán nhãn riêng. Dùng Escalation: Chỉ áp dụng khi vùng cành lá khô hoặc quai chảo kim loại tối màu hòa lẫn hoàn toàn vào bóng tối của góc bếp đến mức mắt người không thể xác định đâu là vật thể, đâu là tường nền. Khi đó, annotator cần gắn cờ hỏi Lab Coach để thống nhất quyết định có gán nhãn hay bỏ qua.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất cho toàn bức ảnh (class_id / class_name). | Ảnh traffic có nhiều đối tượng đồng thời (xe buýt xanh, ô tô con, người đi bộ); mô hình chỉ đoán cab (0.51) và bỏ qua các đối tượng nổi bật khác. | Đọc kỹ guideline quy định đối tượng ưu tiên (diện tích lớn nhất / nằm ở trung tâm) để chọn đúng 1 nhãn duy nhất đại diện cho bức ảnh. | Kiểm tra nhãn được chọn có phản ánh đúng chủ thể chính theo tiêu chí đã thống nhất trong guideline không. |
| Phát hiện vật thể | Tập hợp các bounding box chuẩn hóa: [class_id, x_center, y_center, w, h]. | Trên ảnh kitchen, mô hình nhận nhầm bàn tay trái ở mép thành person 0.61; nhận diện vùng bếp lò củi cổ (đang có than hồng đỏ lửa và thức ăn đang nấu bên trên) thành oven 0.69 do COCO-80 thiếu nhãn bếp nấu (stove); bỏ sót nhiều chảo treo trên cao. | Dùng công cụ vẽ hộp bao ôm sát từng thực thể nhìn thấy được theo danh mục taxonomy COCO-80 và quy ước của guideline. | Soát lỗi False Positives (xóa các box tay/chân bị cắt mép), kiểm tra độ khít của bounding box và đối chiếu xem các vật thể có bị gán ép nhãn (như bếp lò củi thành oven) hay không. |
| Instance segmentation | Tập hợp các chuỗi tọa độ đa giác polygon khép kín cho từng cá thể: [class_id, instance_id, polygon_coords]. | Trên ảnh kitchen, mask của dining table 0.58 bị dính liền và đè trùm lên các khay bánh/bát bột; bó lá khô treo góc trái bị nhòe biên vào tường tối và bị đoán thành potted plant 0.63. | Dùng công cụ đa giác chấm điểm men theo viền thực thể, tách biệt rõ từng instance và khoét rỗng các phần bị che khuất. | Phóng to kiểm tra độ chính xác của viền mask, đảm bảo viền không bị răng cưa cẩu thả, không lem sang nền và không đè lấn các instance khác. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) Chỉ sử dụng ba hình ảnh công khai COCO được cung cấp sẵn trong bài thực hành, tuyệt đối không tải lên hình ảnh cá nhân, ảnh khuôn mặt riêng tư hoặc dữ liệu nội bộ chưa được cấp phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Dừng thao tác ngay lập tức, không tiếp tục gán nhãn hay chạy mô hình và báo cáo ngay cho Lab Coach hoặc Giảng viên phụ trách hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
