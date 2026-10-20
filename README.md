# Temporal_Hallucinate_VideoLLM: server GPU trên Kaggle để nghiên cứu video LM hiểu tính thời gian

Câu hỏi nghiên cứu: **giúp video LM (đặc biệt là mô hình omni có cả tiếng và hình) thực sự hiểu tính thời gian trong video**: sự kiện xảy ra khi nào, trước hay sau, có đồng thời không, và hiểu đúng dù đổi cách hỏi.
Ràng buộc phương pháp: **không huấn luyện, tác động vào bên trong mô hình** (attention, trạng thái ẩn, định tuyến nội tại). Không dùng công cụ ngoài, agent hay ASR.

Repo này là điểm khởi đầu sạch: **server GPU trên Kaggle** giữ sẵn mô hình **Qwen2.5-Omni-3B (Thinker)** và nhận job Python từ máy của bạn qua **ngrok**. Cách làm giống repo `semantic_videoLM`, phần helper viết cho bài toán temporal.

- Mô hình: `Qwen/Qwen2.5-Omni-3B`, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, fp16, 2 GPU T4.
- Dữ liệu mặc định: **SVHalluc temporal** (HF `zcs15/SVHalluc`, revision `211b4f14ab66ae9557b69e302d1f090e0e610839`), 983 câu:

| Task | Tên đầy đủ | Số câu | Đáp án | Câu hỏi mẫu |
|---|---|---|---|---|
| TA | Temporal Alignment | 368 | A/B cân bằng (184/184) | When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No |
| TF | Temporal Forecasting | 276 | A/B/C cân bằng (92 mỗi loại) | Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future |
| CMTB | Cross-Modal Temporal Binding | 339 | A/B/C cân bằng (113 mỗi loại) | When the speech is heard, what visual action is occurring in the video? (A) … (B) … (C) … |

**SVHalluc chỉ là một testbed.** Nó đo việc gắn nội dung lời nói với hình tại khoảnh khắc nói (một phần là ảo giác do lời nói), chưa đủ để kết luận "hiểu thời gian". Các benchmark nên thêm khi có dữ liệu: Daily-Omni (AV Alignment, Event Sequence), FAVE (CVPR 2026), AVTrace (dự kiến công bố tháng 11/2026), và nhóm chỉ hình (TempCompass, TVBench, AoTBench) để kiểm tra không gây hại. Repo này **chưa có bộ đọc dữ liệu cho các benchmark đó**. Muốn thêm thì viết một hàm trả về các `(idx, row)` giống `temporal_rows()` trong `server/temporal_lab.py`.

---

## 1. Cấu trúc

```
Temporal_Hallucinate_VideoLLM/
├── kaggle_temporal_gpu_server.ipynb   notebook server (upload lên Kaggle; sinh ra bằng build_server_notebook.py)
├── build_server_notebook.py           build lại notebook từ server/*.py
├── remote_client.py                   client trên máy bạn: ping / info / run / status / get / upload / restart
├── remote_config.example.json         mẫu file cấu hình (URL + TOKEN)
├── server/
│   ├── temporal_lab.py                namespace chung: nạp model, đọc dữ liệu, chấm điểm, vòng lặp resume
│   ├── kaggle_gpu_server.py           HTTP server (chỉ nghe 127.0.0.1, ngrok chuyển tới)
│   ├── lab_native_video.py            tiền xử lý video/audio chính thức của Qwen (qwen-omni-utils)
│   └── speech_gate.py                 media_layout: vị trí rope, giây của từng token, mặt nạ token hình/tiếng
├── jobs/
│   ├── smoke.py                       kiểm tra server mới: GPU, model, dữ liệu, 1 câu mỗi ô, hai cách chấm
│   ├── baseline.py                    baseline theo (task, đáp án): av, vision, audio
│   └── make_job.py                    ghép tham số vào đầu job (chia shard cho nhiều server)
└── analysis/
    └── analyze_baseline.py            bảng đúng/sai theo (task, đáp án), accuracy thường và cân bằng, hai cách chấm
```

---

## 2. Dựng server trên Kaggle (từng bước)

**Bước 1. Lấy ngrok authtoken**
1. Đăng ký miễn phí tại https://dashboard.ngrok.com/signup.
2. Vào https://dashboard.ngrok.com/get-started/your-authtoken và bấm **Copy**. Đây là *Authtoken*, không phải *API key*. Giữ kín như mật khẩu.
3. Muốn chạy nhiều server cùng lúc thì mỗi server dùng một tài khoản ngrok riêng (gói miễn phí thường chỉ cho một phiên mỗi tài khoản).

**Bước 2. Tạo notebook trên Kaggle** (nên dùng **tài khoản Kaggle phụ** đã xác minh số điện thoại)
1. **Create → New Notebook → File → Import Notebook**, chọn `kaggle_temporal_gpu_server.ipynb`.
2. **Add-ons → Secrets → Add Secret**: Label `NGROK_AUTHTOKEN` (viết đúng, viết hoa), Value = authtoken, và **bật Attach** cho notebook này.
3. Session options: **Accelerator = GPU T4 ×2**, **Internet = On**.

**Bước 3. Save & Run All**
1. **Save Version → Save & Run All (Commit) → Save**. Có thể đóng tab; notebook chạy nền tối đa 12 giờ.
2. Mở **Versions → bản đang chạy → Logs**. Thứ tự: kiểm tra Internet → cài môi trường (khoảng 5–10 phút) → ghi file → nạp model (3–6 phút) → mở ngrok.
3. Khi xong, log in ra:
   ```
   ======================================================================
     URL   : NgrokTunnel: "https://xxxx-xxxx.ngrok-free.dev" -> "http://localhost:8765"
     TOKEN : <chuỗi ngẫu nhiên dài>
   ======================================================================
   ```
   URL là địa chỉ `https://….ngrok-free.dev` trong ngoặc kép. Cả URL lẫn TOKEN **đổi mỗi lần chạy**.

**Bước 4. Kết nối từ máy của bạn**
1. Ở thư mục gốc repo, tạo `remote_config.json` (bị `.gitignore` chặn, **không bao giờ commit**):
   ```json
   {"url": "https://xxxx-xxxx.ngrok-free.dev", "token": "<TOKEN trong log>"}
   ```
   Nhiều server: `remote_config2.json`, `remote_config3.json`, … rồi chọn bằng biến môi trường `REMOTE_CFG=remote_config2.json`.
2. `python remote_client.py ping` phải in `{"ok": true}`; `python remote_client.py info` cho biết GPU và bộ nhớ.
3. `python remote_client.py run jobs/smoke.py` để kiểm tra toàn bộ đường đi (vài phút).

---

## 3. Chạy job

```bash
python remote_client.py run jobs/smoke.py                       # chờ và in stdout + RESULT
python remote_client.py run my_job.py --nowait                  # chỉ gửi, in job id
python remote_client.py status <job_id>                         # xem tiến độ (stdout mới nhất)
python remote_client.py get baseline_s1.jsonl results/baseline_s1.jsonl   # tải file trong OUT về máy
python remote_client.py upload local.json data/local.json       # -> OUT/uploads/data/local.json
python remote_client.py restart                                 # dừng job đang chạy, khởi động lại server (model nạp lại)
```

**Baseline toàn bộ 983 câu, chia cho hai server:**
```bash
python jobs/make_job.py jobs/baseline.py b1.py PER_CELL=0 SHARD=0,2 OUT_NAME=baseline_s1.jsonl
python jobs/make_job.py jobs/baseline.py b2.py PER_CELL=0 SHARD=1,2 OUT_NAME=baseline_s2.jsonl
python remote_client.py run b1.py --nowait
REMOTE_CFG=remote_config2.json python remote_client.py run b2.py --nowait
# khi xong:
python remote_client.py get baseline_s1.jsonl results/baseline_s1.jsonl
REMOTE_CFG=remote_config2.json python remote_client.py get baseline_s2.jsonl results/baseline_s2.jsonl
python analysis/analyze_baseline.py results/baseline_s1.jsonl results/baseline_s2.jsonl
```
`PER_CELL=0` nghĩa là lấy mọi câu dùng được (bỏ clip dài hơn `MAX_DUR`). Job có **resume theo `idx`**: gửi lại đúng file đó sẽ bỏ qua các câu đã xong. Job tự dừng an toàn trước hạn của phiên. Với `av`, `vision`, `audio` mỗi câu cần khoảng 3 lần xử lý media nên baseline đầy đủ khá lâu; thử `PER_CELL=10` trước. Thời gian mỗi câu chưa được đo trên repo này.

**Viết job mới.** Một job là một file Python chạy trong namespace của `server/temporal_lab.py`, dùng trực tiếp được:

| Tên | Công dụng |
|---|---|
| `MODEL_`, `PROC`, `L4` | mô hình, processor, id token của chữ A B C D |
| `temporal_rows()`, `balanced_items(per_cell, seed)` | dữ liệu; chọn cân bằng theo (task, đáp án), bỏ clip hỏng hoặc quá dài **trước** khi chọn |
| `clip_info(row)` | (đường dẫn, thời lượng, codec audio) |
| `parse_question(q)`, `prompt_of(stem, options, perm)`, `rotations(k)` | tách và dựng câu hỏi |
| `prepare_video(path)` | tiền xử lý chính thức của Qwen (video bằng decord, audio bằng PyAV) |
| `qwen_inputs(PROC, prompt, media, mode)`, `move_inputs(inp, MODEL_)` | `mode` = `av` / `vision` / `audio` |
| `score(media, stem, options, mode, fwd, both)` | log-prob từng phương án, **cộng qua mọi cách xoay thứ tự phương án**; `both=True` trả thêm log-prob với thứ tự gốc (cách chấm của các bảng công bố); `fwd(inputs)` thay forward để can thiệp |
| `media_layout(MODEL_, PROC, inputs)` | vị trí rope, giây của từng token, mặt nạ token hình / tiếng (dùng khi can thiệp attention) |
| `narrated_block(audio)` | đoạn lời nói (giây bắt đầu, giây kết thúc) của một clip SVHalluc, chỉ để phân tích |
| `run_loop(items, out_name, fn, show)` | vòng lặp resume, ghi `OUT/out_name`, in đúng/sai theo (task, đáp án) khi chạy |
| `OUT` | thư mục kết quả trên server |

Đặt `RESULT = {...}` ở cuối job nếu muốn client in ra kết quả tóm tắt. Kết quả lớn thì ghi file vào `OUT` rồi tải bằng `get`.

---

## 4. Lưu ý về dữ liệu và cách chấm (đọc trước khi kết luận gì)

1. **Hai cách chấm, luôn báo cả hai.** `score()` cộng log-prob qua mọi cách xoay thứ tự phương án để loại thiên kiến vị trí chữ cái. Các bảng công bố (ví dụ bảng của RED, arXiv 2610.02976, Qwen2.5-Omni-3B: TA 48,1; TF 21,4; CMTB 49,0) báo accuracy thường với thứ tự gốc. `both=True` cho cả hai từ cùng các lần chạy mô hình; chỉ so với bảng công bố bằng phiên bản `_canon`.
2. **Không rò rỉ codec ở temporal.** 164/164 clip temporal đã kiểm tra đều có audio AAC (khác với SVHalluc semantic, nơi clip đáp án No dùng pcm và clip đáp án Yes dùng AAC).
3. **Cấu trúc audio.** Audio của một clip là im lặng số cộng **một khối lời nói liên tục** với ranh giới tròn theo giây (trung vị khoảng 13,6 giây trong clip 23–28 giây), gần như không có chỗ ngắt giữa các câu. Điều này giới hạn các phép thử kiểu "dời lời nói".
4. **Thời lượng.** Clip dài hơn `MAX_DUR` (42 s với 2 GPU) bị bỏ **trước** khi chọn mẫu và được liệt kê trong `SKIPPED`.
5. **Báo kết quả theo từng (task, đáp án đúng)**, kèm phân bố đáp án. Độ chính xác gộp che mất hiện tượng mô hình dồn hết về một đáp án (ví dụ TF toàn "Present").
6. **Đối chứng bắt buộc cho mọi claim về thời gian:** bỏ một phương thức (`vision`, `audio`), đảo hoặc dời thời gian. Một bài 2026 cho thấy nhiều điểm "temporal" đến từ một khung hình đơn và thiên kiến đáp án.

---

## 5. Lỗi thường gặp

| Triệu chứng | Cách sửa |
|---|---|
| `Không kết nối được … Bật Internet` | Session options → Internet = **On** |
| `Không đọc được Kaggle Secret NGROK_AUTHTOKEN` | Chưa thêm Secret, sai tên, hoặc chưa bật **Attach** |
| ngrok báo authtoken sai / đang có phiên khác | Dùng *Authtoken* (không phải API key); tài khoản ngrok đó đang mở server khác → dùng tài khoản khác |
| `server không lên được` | Xem đuôi `server.log` được in ra; thường là hết bộ nhớ GPU hoặc thiếu GPU thứ hai |
| `{"error": "bad token"}` / HTTP 401 | TOKEN trong `remote_config.json` không khớp lần chạy hiện tại |
| Không kết nối tới URL (ngrok 3200) | Phiên đã hết giờ hoặc bị ngắt → chạy lại Bước 3, cập nhật URL + TOKEN |
| Job treo | `python remote_client.py restart` (tải kết quả dang dở về trước) |

---

## 6. Bảo mật

- **Ai có cả URL lẫn TOKEN đều chạy được mã tuỳ ý trong notebook của bạn.** Không đăng lên GitHub, không gửi vào kênh công khai, không chia sẻ log của notebook.
- URL/TOKEN chỉ ghi trong `remote_config*.json`, file này đã bị `.gitignore` chặn. Kiểm tra lại bằng `git status` trước mỗi lần commit.
- Server chỉ nghe `127.0.0.1`; token được so sánh theo thời gian hằng số; `/file` chỉ đọc trong thư mục kết quả; `/upload` chỉ ghi vào `OUT/uploads`.
- Notebook cố ý không có vòng lặp chống ngắt. Dùng tài khoản phụ và tải kết quả về thường xuyên. Quota GPU của Kaggle tính theo tuần.
