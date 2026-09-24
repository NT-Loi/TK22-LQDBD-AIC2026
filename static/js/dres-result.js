const VERDICT_PRESENTATION = {
  CORRECT: { label: "ĐÚNG", kind: "correct" },
  WRONG: { label: "SAI", kind: "wrong" },
  INDETERMINATE: { label: "CHƯA XÁC ĐỊNH", kind: "indeterminate" },
  UNDECIDABLE: { label: "KHÔNG THỂ CHẤM", kind: "undecidable" },
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
}

export function bindCopyButton(button, getText) {
  button?.addEventListener("click", () => copyText(getText(), button));
}
