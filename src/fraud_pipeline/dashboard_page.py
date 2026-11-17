"""HTML shell and local chat UI for the Plotly dashboard."""


def render_dashboard(plot_html: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fraud Pipeline Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1a3039;
      --muted: #657982;
      --line: #dce5e6;
      --canvas: #f3f6f6;
      --surface: #ffffff;
      --teal: #0e766e;
      --teal-soft: #e5f2ef;
      --coral: #b94d3b;
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; min-height: 100%; margin: 0; }
    body {
      background: var(--canvas);
      color: var(--ink);
      font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
    }
    .dashboard-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
      min-height: 720px;
    }
    .visualization { min-width: 0; overflow: auto; padding: 18px; }
    .plotly-graph-div { width: 100% !important; }
    .assistant-panel {
      display: flex;
      min-height: 0;
      flex-direction: column;
      border-left: 1px solid var(--line);
      background: var(--surface);
    }
    .assistant-header { padding: 22px 20px 18px; border-bottom: 1px solid var(--line); }
    .header-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .eyebrow { margin: 0 0 5px; color: var(--teal); font-size: 11px; font-weight: 700; }
    h1 { margin: 0; font-size: 19px; line-height: 1.25; }
    .local-status {
      flex: 0 0 auto;
      padding: 5px 8px;
      border: 1px solid #cfe4df;
      border-radius: 4px;
      background: var(--teal-soft);
      color: var(--teal);
      font-size: 10px;
      font-weight: 700;
    }
    .conversation { flex: 1; min-height: 0; overflow-y: auto; padding: 18px 16px; }
    .message { width: fit-content; max-width: 94%; margin: 0 0 12px; padding: 11px 12px; border-radius: 8px; font-size: 13px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message.user { margin-left: auto; background: var(--teal); color: white; }
    .message.assistant { border: 1px solid var(--line); background: #f8faf9; color: var(--ink); }
    .message.error { border-color: #efc9c1; color: var(--coral); }
    .suggestions { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 16px 14px; }
    .suggestion {
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 7px 9px;
      background: white;
      color: var(--ink);
      font: inherit;
      font-size: 11px;
      cursor: pointer;
    }
    .suggestion:hover { border-color: var(--teal); color: var(--teal); }
    .chat-form { display: flex; align-items: flex-end; gap: 8px; padding: 14px 16px; border-top: 1px solid var(--line); }
    .chat-input {
      flex: 1;
      min-width: 0;
      min-height: 42px;
      max-height: 120px;
      resize: vertical;
      border: 1px solid #cbd7d9;
      border-radius: 5px;
      padding: 10px;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }
    .chat-input:focus { outline: 2px solid #a9d2cc; border-color: var(--teal); }
    .send-button {
      min-height: 42px;
      border: 0;
      border-radius: 5px;
      padding: 0 14px;
      background: var(--teal);
      color: white;
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .send-button:disabled { opacity: 0.55; cursor: wait; }
    .panel-footer { padding: 0 16px 14px; color: var(--muted); font-size: 10px; }
    @media (max-width: 820px) {
      .dashboard-layout { display: flex; height: auto; min-height: 100vh; flex-direction: column; }
      .visualization { overflow-x: auto; padding: 8px; }
      .visualization .plotly-graph-div { width: 720px !important; min-width: 720px !important; }
      .assistant-panel { min-height: 620px; border-top: 1px solid var(--line); border-left: 0; }
    }
  </style>
</head>
<body>
  <div class="dashboard-layout">
    <main class="visualization">{{plot}}</main>
    <aside class="assistant-panel" aria-label="Fraud pipeline assistant">
      <header class="assistant-header">
        <div class="header-row">
          <div><p class="eyebrow">PIPELINE ANALYSIS</p><h1>Fraud assistant</h1></div>
          <span class="local-status">LOCAL</span>
        </div>
      </header>
      <section id="conversation" class="conversation" aria-live="polite"></section>
      <nav class="suggestions" aria-label="Report questions">
        <button class="suggestion" data-prompt="Summarize this pipeline run.">Run summary</button>
        <button class="suggestion" data-prompt="Which features show the most drift, and what does it mean?">Explain drift</button>
        <button class="suggestion" data-prompt="Did any data quality checks fail?">Check data quality</button>
      </nav>
      <form id="chat-form" class="chat-form">
        <textarea id="chat-input" class="chat-input" rows="2" maxlength="4000" placeholder="Ask about this run…" aria-label="Message"></textarea>
        <button id="send-button" class="send-button" type="submit">Send</button>
      </form>
      <footer class="panel-footer">Read-only reports · local model</footer>
    </aside>
  </div>
  <script>
    const conversation = document.getElementById("conversation");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const sendButton = document.getElementById("send-button");
    const history = [];

    function addMessage(role, content, isError = false) {
      const message = document.createElement("div");
      message.className = `message ${role}${isError ? " error" : ""}`;
      message.textContent = content;
      conversation.appendChild(message);
      conversation.scrollTop = conversation.scrollHeight;
    }

    async function sendMessage(content) {
      const text = content.trim();
      if (!text || sendButton.disabled) return;
      history.push({ role: "user", content: text });
      addMessage("user", text);
      input.value = "";
      sendButton.disabled = true;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: history })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Request failed.");
        history.push({ role: "assistant", content: result.answer });
        addMessage("assistant", result.answer);
      } catch (error) {
        addMessage("assistant", error.message, true);
      } finally {
        sendButton.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      sendMessage(input.value);
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    document.querySelectorAll(".suggestion").forEach(button => {
      button.addEventListener("click", () => sendMessage(button.dataset.prompt));
    });
  </script>
</body>
</html>""".replace("{{plot}}", plot_html)