# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

1. `frame_0182.jpg` — Điểm: 0.9591 | Thời điểm: 72.8s | Thứ tự (Rank): 1
   - _Lý do:_ Là frame có điểm số cao nhất toàn tập pool (rank 1), số box mơ hồ cao nhất ($A = 1.0$, 18 box mơ hồ trong 28 box dự đoán), độ bất định lớn ($U = 0.9182$). Đem lại lượng thông tin phản hồi lớn nhất cho model.
2. `frame_0369.jpg` — Điểm: 0.9324 | Thời điểm: 147.6s | Thứ tự (Rank): 2
   - _Lý do:_ Rank 2 toàn pool với mật độ xe cao (43 box, 16 box mơ hồ, $U = 0.9315$), biểu thị giao thông dày đặc ở nửa sau video, giúp bổ sung nhiều mẫu khó về mật độ xe.
3. `frame_0326.jpg` — Điểm: 0.9155 | Thời điểm: 130.4s | Thứ tự (Rank): 4
   - _Lý do:_ Điểm rất cao ($U = 0.9310$, $A = 0.8333$), có 39 box. Đặc biệt, việc chọn frame này giúp ta loại trừ `frame_0331.jpg`\*\* (rank 5, t = 132.4s) vì hai ảnh chỉ cách nhau đúng 2.0s, tránh lãng phí ngân sách rà nhãn vào cảnh gần trùng lặp.
4. `frame_0099.jpg` — Điểm: 0.9063 | Thời điểm: 39.6s | Thứ tự (Rank): 8
   - _Lý do:_ Đại diện quan trọng cho giai đoạn đầu video (t = 39.6s), có độ bất định dự đoán rất cao ($U = 0.9460$, 14 box mơ hồ). Đảm bảo tính phân tán theo dòng thời gian thay vì dồn hết vào cuối video.
5. `frame_0270.jpg` — Điểm: 0.8878 | Thời điểm: 108.0s | Thứ tự (Rank): 13
   - _Lý do:_ Nằm ở khoảng giữa dòng thời gian (t = 108.0s), cách xa các cụm ảnh khác ($D = 1.0$), có $U = 0.9089$. Dù điểm thấp hơn rank 3 (`frame_0380.jpg`) nhưng chọn frame này giúp phân bổ đều dữ liệu đa dạng trong toàn bộ video thay vì tập trung quá dày đặc ở dải $145s - 155s$.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: frame_0099.jpg, frame_0270.jpg, frame_0392.jpg

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

- frame_0331.jpg (Rank 5, Score 0.9154, t = 132.4s): Có điểm rất cao nhưng không nên ưu tiên nếu giới hạn ngân sách 5 ảnh, vì nó nằm quá gần `frame_0326.jpg` (t = 130.4s, chỉ cách 2.0s). Bối cảnh giao thông và vị trí các xe hầu như chưa thay đổi đáng kể, việc gán nhãn cả hai sẽ gây trùng lặp thông tin và lãng phí công sức gán nhãn.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

- Điểm chọn cao chỉ phản ánh sự lúng túng/bất định (uncertainty) của mô hình hiện tại đối với phân phối dữ liệu của frame đó, chứ không đảm bảo rằng khi gán nhãn xong và huấn luyện lại thì mô hình chắc chắn sẽ tổng quát hóa tốt hơn.
- Nếu dữ liệu được chọn chứa quá nhiều nhiễu (xe quá mờ/nhỏ sát đường chân trời, đèn lóa quá nặng), hoặc phân phối nhãn gán tay bị lệch (ví dụ người gán chỉ thêm xe to/vừa mà bỏ qua xe nhỏ, hoặc số lượng box trùng lặp cao), mô hình có thể bị overfitting hoặc học phải tín hiệu nhiễu, dẫn đến việc sụt giảm độ phủ (recall) và AP50 trên tập kiểm thử độc lập.
