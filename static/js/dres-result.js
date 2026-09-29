const VERDICT_PRESENTATION = {
  CORRECT: { label: "ĐÚNG", kind: "correct" },
  WRONG: { label: "SAI", kind: "wrong" },
  INDETERMINATE: { label: "CHƯA XÁC ĐỊNH", kind: "indeterminate" },
  UNDECIDABLE: { label: "KHÔNG THỂ CHẤM", kind: "undecidable" },
  PARTIALLY_CORRECT: { label: "ĐÚNG MỘT PHẦN", kind: "indeterminate" },
};

function addRow(container, label, value) {
  if (value === undefined || value === null || value === "") return;
  const row = document.createElement("div");
  row.className = "dres-result-row";
  const name = document.createElement("strong");
  name.textContent = `${label}:`;
  const content = document.createElement("span");
  content.textContent = String(value);
  row.append(name, content);
  container.appendChild(row);
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = "Đã sao chép";
    setTimeout(() => { button.textContent = original; }, 1200);
  } catch {
    button.textContent = "Không thể sao chép";
  }
}

export function clearDresResult(container) {
  if (!container) return;
  container.replaceChildren();
  container.className = "submission-result";
}

export function renderDresReceipt(container, receipt) {
  if (!container) return;
  clearDresResult(container);

  const presentation = receipt.pending
    ? { label: "ĐANG CHỜ VERDICT", kind: "pending" }
    : (VERDICT_PRESENTATION[receipt.verdict] || { label: "KHÔNG RÕ", kind: "indeterminate" });

  container.classList.add("submission-result-card", presentation.kind);
  const title = document.createElement("h4");
  title.textContent = `DRES ĐÃ NHẬN • ${presentation.label}`;
  container.appendChild(title);

  addRow(container, "HTTP", receipt.dresHttpStatus ?? receipt.dresStatus);
  addRow(container, "Verdict", receipt.pending ? "PENDING" : receipt.verdict);
  addRow(container, "Mô tả", receipt.description);
  addRow(container, "Evaluation", receipt.evaluationId);
  addRow(container, "Đích", receipt.destination);

  const raw = receipt.rawResult ?? receipt.result;
  if (raw !== undefined) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Phản hồi JSON từ DRES";
    const pre = document.createElement("pre");
    const rawText = JSON.stringify(raw, null, 2);
    pre.textContent = rawText;
    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "dres-copy-btn";
    copyButton.textContent = "Sao chép phản hồi";
    copyButton.addEventListener("click", () => copyText(rawText, copyButton));
    details.append(summary, pre, copyButton);
    container.appendChild(details);
  }
}

export function renderDresError(container, error) {
  if (!container) return;
  clearDresResult(container);
  container.classList.add("submission-result-card", "error");
  const title = document.createElement("h4");
  title.textContent = error.status === 412
    ? "DRES TỪ CHỐI SUBMISSION"
    : "KHÔNG NỘP ĐƯỢC BÀI";
  container.appendChild(title);
  addRow(container, "HTTP", error.status || "Không kết nối được");
  addRow(container, "Lý do", error.message);

  let hint = null;
  const msg = String(error.message || "").toLowerCase();
  if (msg.includes("not running")) {
    hint = "⚠️ Task này hiện chưa được BTC kích hoạt trên DRES (hoặc đã kết thúc). Nếu cuộc thi có nhiều kỳ đánh giá (ví dụ Session 2), hãy nhấn nút 'Đổi Evaluation' ở thanh DRES để chuyển sang kỳ đang chạy.";
  } else if (msg.includes("maximum number of correct submissions")) {
    hint = "✅ Đội đã nộp đúng câu này trước đó và đã đạt giới hạn tối đa của BTC cho task này.";
  } else if (msg.includes("could not find media item")) {
    hint = "⚠️ Tên video không khớp với danh mục trên DRES. Hệ thống đã tự động chuẩn hóa định dạng (Nxxx-Vyyy / Lxx_Vyyy).";
  }
  if (hint) {
    const hintDiv = document.createElement("div");
    hintDiv.className = "dres-hint-box";
    hintDiv.style.marginTop = "8px";
    hintDiv.style.padding = "8px 12px";
    hintDiv.style.background = "rgba(227, 179, 65, 0.15)";
    hintDiv.style.border = "1px solid #e3b341";
    hintDiv.style.color = "#ffd33d";
    hintDiv.style.borderRadius = "6px";
    hintDiv.style.fontSize = "12px";
    hintDiv.style.lineHeight = "1.4";
    hintDiv.textContent = hint;
    container.appendChild(hintDiv);
  }
}

export function bindCopyButton(button, getText) {
  button?.addEventListener("click", () => copyText(getText(), button));
}
