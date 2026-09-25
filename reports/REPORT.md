# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Ngọc Nguyên

Công cụ gán nhãn đã dùng: Cvat(AnyLabeling, CVAT, SAM hoặc sửa trực tiếp file nhãn)

Sao chép file này thành `reports/REPORT.md` rồi điền vào các chỗ . Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

- Lý do chia theo trục thời gian có vùng đệm:
  Dữ liệu video có tính tương quan thời gian (temporal correlation) cực kỳ cao giữa các frame liền kề (vị trí xe, điều kiện ánh sáng, góc quay camera, hướng di chuyển gần như giống hệt nhau). Việc chia theo trục thời gian và đặt một vùng đệm (buffer zone) ở giữa đảm bảo tập kiểm thử (test set) độc lập hoàn toàn với tập huấn luyện/pool về mặt chuỗi thời gian, phản ánh đúng khả năng tổng quát hóa (generalization) của mô hình trên các thời điểm chưa từng thấy.
- Nếu chia ngẫu nhiên:
  Số đo trên tập kiểm thử sẽ bị lệch lạc theo hướng lạc quan giả tạo (overly optimistic / data leakage), nghĩa là các chỉ số như AP50, Precision, Recall sẽ cao bất thường. Nguyên nhân là do hiện tượng rò rỉ dữ liệu (data leakage): một frame ở tập test sẽ gần như trùng khớp với một frame ở tập train/pool chỉ cách nhau vài phần mười giây, khiến mô hình chỉ cần "học vẹt" bối cảnh là đã phát hiện chính xác mà không thực sự học được đặc trưng tổng quát.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

- Bảng số liệu Vòng 0 (Cold Start):

| Vòng | Ảnh train | Box train |  AP50  | ΔAP50 |   P    |   R    |   F1   | R (small) | R (medium) | R (large) |
| :--: | :-------: | :-------: | :----: | :---: | :----: | :----: | :----: | :-------: | :--------: | :-------: |
|  0   |     0     |     0     | 0.7714 |   -   | 0.9249 | 0.4888 | 0.6396 |  0.1818   |   0.5473   |  0.5610   |

- Đánh giá dựa trên `compare_round0.jpg` và kích thước xe:\*\*
  - Mô hình khởi đầu lạnh (`yolov8n` pretrained trên COCO) nhận diện rất tốt các xe kích thước lớn và vừa ở cự ly gần đến trung bình (Precision đạt 0.9249). Tuy nhiên, mô hình bỏ sót rất nhiều (FN - box vàng) ở các xe nhỏ ở xa, xe ở làn ngược chiều bị đèn pha chiếu chói lóa, và các xe bị che khuất một phần (occlusion).
  - Độ phủ (Recall) theo kích thước xe: Độ phủ đối với xe nhỏ (`small`) cực kỳ thấp, chỉ đạt 18.18% (bỏ sót hơn 81% số lượng xe nhỏ), trong khi xe vừa (`medium`) đạt 54.73% và xe lớn (`large`) đạt 56.10%. Điều này cho thấy COCO pretrained chưa thích nghi tốt với các vật thể nhỏ xíu bị lóa sáng ban đêm sát đường chân trời.
  - Trường hợp cần người rà lại nhãn tham chiếu: Khi mô hình dự đoán đúng một xe thực tế nhưng nhãn tham chiếu (vốn được tạo bán tự động) lại bỏ sót (False Positive giả của mô hình), hoặc các đốm sáng đèn pha/biển báo phản quang bị nhãn tham chiếu gán nhầm thành xe. Người chấm cần trực tiếp rà lại nhãn tham chiếu trước khi kết luận mô hình sai.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

- Giải thích công thức và tham số:
  - Công thức tính điểm chọn mẫu kết hợp 3 thành phần có trọng số ($W_U = 0.5, W_A = 0.3, W_D = 0.2$):
    - $U$ (Uncertainty): Độ bất định trung bình của các dự đoán trong frame (dựa trên entropy hoặc độ phân vân của confidence score). Điểm càng cao, mô hình càng lúng túng.
    - $A$ (Ambiguity): Tỷ lệ/số lượng các bounding box nằm trong dải ngưỡng không chắc chắn (conf xung quanh 0.25 - 0.5), đo lường mức độ khó của các vật thể trong frame.
    - $D$ (Diversity): Độ đa dạng và phân tán về mặt thời gian so với các frame đã chọn trước đó, giúp tránh gom cụm.
  - Vai trò của `MIN_GAP_S` (= 2.0s): Là khoảng cách thời gian tối thiểu giữa hai frame liên tiếp trong cùng một batch. Nó ngăn thuật toán chọn các frame liền kề gần như giống nhau (redundant frames), buộc thuật toán phải chọn mẫu rải đều theo chuỗi thời gian để tối ưu hóa ngân sách gán nhãn.
- Chứng minh qua các frame cụ thể:
  - `frame_0182.jpg` (Rank 1, t = 72.8s, Score = 0.9591): Có độ bất định cao ($U = 0.9182$) và độ mơ hồ tuyệt đối ($A = 1.0$), là mẫu giàu thông tin nhất.
  - `frame_0369.jpg` (Rank 2, t = 147.6s, Score = 0.9324) và `frame_0326.jpg` (Rank 4, t = 130.4s, Score = 0.9155): Đại diện cho các khung cảnh mật độ xe đông đúc ở nửa sau video.
  - `frame_0331.jpg` (Rank 5, t = 132.4s, Score = 0.9154): Dù có điểm rất cao nhưng nằm sát `frame_0326.jpg` (chênh 2.0s), cho thấy sự cần thiết của việc cân nhắc ảnh gần trùng để tiết kiệm công gán nhãn.
  - `frame_0099.jpg` (Rank 8, t = 39.6s, Score = 0.9063, $U = 0.9460$): Được chọn để đảm bảo tính bao phủ ở giai đoạn đầu video.
- Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Vì sao?
  - Không. Điểm bất định chỉ cho biết mô hình hiện tại đang thiếu tự tin đối với ảnh đó. Nếu frame bất định do chất lượng ảnh quá kém (nhiễu hạt, lóa đèn nặng, vật thể không rõ ràng) hoặc nhãn gán tay đưa vào bị mâu thuẫn/lệch phân phối so với tập test, việc nạp thêm ảnh này chỉ làm mô hình bị nhiễu (noisy gradients), dẫn tới giảm độ khái quát hóa thay vì cải thiện.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

- Bảng tổng hợp các vòng (`rounds_table.md`):

| Vòng | Mô hình                 | Ảnh train | Box train |  AP50  |  ΔAP50  |   P    |   R    |   F1   | R (small) | R (medium) | R (large) |
| :--: | :---------------------- | :-------: | :-------: | :----: | :-----: | :----: | :----: | :----: | :-------: | :--------: | :-------: |
|  0   | yolov8n cold start      |     0     |     0     | 0.7714 |    -    | 0.9249 | 0.4888 | 0.6396 |  0.1818   |   0.5473   |  0.5610   |
|  1   | yolov8n fine-tune v1..1 |    12     |    336    | 0.5477 | -0.2237 | 1.0000 | 0.1886 | 0.3173 |  0.0000   |   0.2027   |  0.3902   |

- Mức độ chỉnh sửa nhãn gợi ý ở Vòng 1 (từ `round1_diff.md`):
  - Số box pre-label do mô hình đề xuất: 169 box.
  - Số box giữ nguyên (accepted): 109 box (tỷ lệ chấp nhận 64.5%).
  - Số box chỉnh sửa tọa độ (edited): 42 box.
  - Số box xóa bỏ (deleted - False Positive của pre-label): 18 box.
  - Số box gán thêm mới (added - False Negative của pre-label): 185 box.
  - Tổng số box sau khi rà soát và đóng gói: 336 box trên 12 ảnh.
- Biến động số đo:
  - AP50: Giảm mạnh từ 0.7714 xuống 0.5477
  - Precision: Tăng tuyệt đối lên 1.0000 (không có bất kỳ False Positive nào trên ngưỡng conf 0.25).
  - Recall: Giảm nghiêm trọng từ 0.4888 xuống 0.1886 (bỏ sót tới 327/403 box tham chiếu).
  - Theo kích thước xe: Xe nhỏ (`small`) giảm recall về 0.0000 (không nhận diện được bất kỳ xe nhỏ nào), xe vừa (`medium`) giảm từ 54.73% xuống 20.27%, xe lớn (`large`) giảm từ 56.10% xuống 39.02%.
- Chỉ ra ca kết quả thay đổi sau fine-tune:
  - _Kết quả xấu đi:_ Mô hình sau fine-tune trở nên cực kỳ "thận trọng" (conservative), độ tự tin giảm sâu dẫn đến việc bỏ sót hàng loạt xe ở cự ly xa và xe vừa. Cụ thể, trong `frame_0392.jpg` khi train đã bị phát hiện thiếu và có 15 duplicate labels bị loại bỏ; mô hình học trên tập 12 ảnh quá nhỏ bị overfit vào nền tối và chỉ phát hiện những xe cực kỳ rõ ràng ở gần.
- Phân biệt ba khái niệm:
  - _Quan sát độc lập (Blind Scan):_ Việc người gán nhãn tự dò tìm và xác định các vị trí có xe trên ảnh mà không nhìn vào bounding box do AI gợi ý trước, tránh bị thiên kiến xác nhận (confirmation bias).
  - _Lỗi pre-label đã sửa (round1_diff):_ Sai lệch khách quan của mô hình khởi đầu lạnh so với nhãn chuẩn sau khi người rà soát (xóa 18 box sai, thêm 185 box thiếu).
  - _Kết quả mô hình sau train:_ Khả năng suy luận thực tế của trọng số mới (`last.pt`) trên 20 ảnh test hoàn toàn độc lập.
- Ca khó theo guideline: Hai xe đi sát nhau ở làn ngược chiều bị đèn pha đối diện chiếu thẳng vào camera tạo thành một quầng sáng trắng xóa duy nhất; pre-label gộp chung thành một box to hoặc bỏ qua, người rà nhãn phải tách thành hai box dựa vào vệt sáng gầm xe và nóc xe.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

- Kết quả Vòng 1 so với Cold Start:
  Số đo AP50 và Recall bị tụt giảm nghiêm trọng (AP50 giảm từ 77.1% xuống 54.8%, Recall giảm từ 48.9% xuống 18.9%) mặc dù Precision đạt 100%. Đây là hiện tượng phổ biến ở vòng đầu của active learning khi số lượng ảnh huấn luyện còn quá ít (chỉ 12 ảnh) và mô hình bị phạt nặng về phân phối xác suất phân lớp khi chuyển từ 80 lớp COCO về 1 lớp duy nhất.
- Quyết định: Tiếp tục vòng 2. Chưa thể dừng lại vì mô hình mới chỉ học 12 ảnh, chưa đạt tới điểm bão hòa dữ liệu và cần thêm các mẫu đa dạng để khôi phục lại Recall đối với xe nhỏ và xe tầm trung.
- Đề xuất hai ca cho vòng sau:
  1. _Cụm xe nhỏ ở cự ly xa gần đường chân trời (bị lóa đèn):_ Chi phí rà nhãn cao (mỗi ảnh có thể chứa 35 - 45 box nhỏ, tốn nhiều công zoom/vẽ box tỉ mỉ), nguy cơ gần trùng trung bình.
  2. _Xe tải và xe buýt chạy ở làn ngoài cùng sát rào chắn:_ Chi phí rà nhãn thấp/trung bình (box to, dễ vẽ), nguy cơ gần trùng thấp nếu chọn các frame cách nhau trên 10 giây.
- Ảnh hưởng từ giới hạn của tập kiểm thử:
  - Tập test chỉ có 20 ảnh khiến phương sai thống kê lớn; một vài box bị bỏ sót có thể làm biến động mạnh chỉ số Recall và AP50.
  - Luật bỏ qua xe cao dưới 16 pixel giúp tránh phạt mô hình ở những trường hợp bất khả thi về mặt quang học, nhưng nhãn tham chiếu do mô hình tạo chưa được rà thủ công 100% có thể chứa lỗi hệ thống, khiến việc đánh giá mô hình học chủ động bị lệch chuẩn.
- Nếu AP50 giảm, những điều cần kiểm tra trước khi train thêm:
  1. Kiểm tra chất lượng và độ nhất quán của file nhãn vừa sửa: có bị trùng lặp nhãn (như lỗi 15 duplicate labels ở `frame_0392.jpg`), box bị lệch tọa độ, hoặc người gán nhãn quên không gán xe nhỏ hay không.
  2. Kiểm tra ngưỡng tin cậy (`conf_threshold`): sau khi fine-tune, phân phối confidence của model thường bị co lại; nếu vẫn giữ ngưỡng conf 0.25 để tính AP/Recall thì nhiều dự đoán đúng ở mức conf 0.15 - 0.24 sẽ bị tính là False Negative.
  3. Kiểm tra siêu tham số huấn luyện: learning rate (lr0), số
