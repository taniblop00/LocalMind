"""
LocalMind — Upload Server
Run this BEFORE main.py when working on a remote server.
It starts a web interface on port 7860 where you can
drag & drop your training documents into data/raw/.

Usage:
    python upload_server.py
Then open the printed URL in your browser.
"""

import os
import http.server
import socketserver
import cgi
import json
from pathlib import Path
from urllib.parse import urlparse

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}

PORT = 7860  # Same port as Gradio/HuggingFace Spaces, usually open on Vast.ai

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LocalMind — File Upload</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }

  .container {
    width: 100%;
    max-width: 640px;
  }

  .logo {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .logo h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
  }

  .logo p {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.4rem;
  }

  .drop-zone {
    border: 2px dashed #334155;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: #0f1117;
    position: relative;
  }

  .drop-zone:hover, .drop-zone.dragover {
    border-color: #a855f7;
    background: #12101d;
    transform: scale(1.01);
  }

  .drop-zone .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
  }

  .drop-zone h2 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .drop-zone p {
    color: #64748b;
    font-size: 0.85rem;
  }

  .formats {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.2rem;
  }

  .format-tag {
    background: #1e293b;
    color: #94a3b8;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: monospace;
  }

  #file-input { display: none; }

  .upload-btn {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.6rem 1.5rem;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
  }

  .upload-btn:hover { opacity: 0.85; }

  .file-list {
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .file-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #0f1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .file-item .file-icon { font-size: 1.4rem; }

  .file-info { flex: 1; min-width: 0; }

  .file-name {
    font-size: 0.88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .file-size {
    font-size: 0.75rem;
    color: #64748b;
  }

  .file-status {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
  }

  .status-uploading { background: #1e3a5f; color: #60a5fa; }
  .status-done      { background: #14532d; color: #4ade80; }
  .status-error     { background: #450a0a; color: #f87171; }

  .progress-bar-wrap {
    height: 3px;
    background: #1e293b;
    border-radius: 99px;
    margin-top: 0.4rem;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #a855f7, #3b82f6);
    border-radius: 99px;
    transition: width 0.2s;
    width: 0%;
  }

  .existing-files {
    margin-top: 2rem;
    background: #0f1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
  }

  .existing-files h3 {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
  }

  .existing-file-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.85rem;
  }

  .existing-file-row:last-child { border-bottom: none; }

  .existing-name { color: #94a3b8; }
  .existing-size { color: #475569; font-size: 0.78rem; }

  .ready-banner {
    margin-top: 1.5rem;
    padding: 1rem 1.2rem;
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 10px;
    font-size: 0.88rem;
    color: #4ade80;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .footer {
    margin-top: 2.5rem;
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
  }

  .footer code {
    background: #1e293b;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    color: #94a3b8;
  }
</style>
</head>
<body>
<div class="container">

  <div class="logo">
    <h1>🧠 LocalMind</h1>
    <p>Upload your training documents to <strong>data/raw/</strong></p>
  </div>

  <div class="drop-zone" id="drop-zone">
    <span class="icon">📂</span>
    <h2>Drag & Drop your files here</h2>
    <p>or click to browse your computer</p>
    <div class="formats">
      <span class="format-tag">.pdf</span>
      <span class="format-tag">.txt</span>
      <span class="format-tag">.csv</span>
      <span class="format-tag">.docx</span>
    </div>
    <button class="upload-btn" onclick="document.getElementById('file-input').click()">Browse Files</button>
  </div>

  <input type="file" id="file-input" multiple accept=".pdf,.txt,.csv,.docx">

  <div class="file-list" id="file-list"></div>

  <div id="ready-banner" style="display:none" class="ready-banner">
    ✅ Files uploaded successfully! Close this tab and run <strong>python main.py</strong>
  </div>

  <div id="existing-container"></div>

  <div class="footer">
    Files are saved to <code>data/raw/</code> on the server.<br>
    When done, run <code>python main.py</code> to start training.
  </div>

</div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
let uploadedCount = 0;

// Load existing files on page load
async function loadExisting() {
  const res = await fetch('/list');
  const data = await res.json();
  if (data.files.length === 0) return;
  const container = document.getElementById('existing-container');
  container.innerHTML = `<div class="existing-files">
    <h3>Already in data/raw/</h3>
    ${data.files.map(f => `<div class="existing-file-row">
      <span class="existing-name">📄 ${f.name}</span>
      <span class="existing-size">${f.size}</span>
    </div>`).join('')}
  </div>`;
}
loadExisting();

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  uploadFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => uploadFiles([...fileInput.files]));

function humanSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function addFileRow(file) {
  const id = 'file-' + Date.now() + Math.random();
  const div = document.createElement('div');
  div.className = 'file-item';
  div.id = id;
  div.innerHTML = `
    <span class="file-icon">📄</span>
    <div class="file-info">
      <div class="file-name">${file.name}</div>
      <div class="file-size">${humanSize(file.size)}</div>
      <div class="progress-bar-wrap"><div class="progress-bar" id="bar-${id}"></div></div>
    </div>
    <span class="file-status status-uploading" id="status-${id}">Uploading…</span>
  `;
  fileList.appendChild(div);
  return id;
}

async function uploadFiles(files) {
  for (const file of files) {
    const rowId = addFileRow(file);
    const bar = document.getElementById('bar-' + rowId);
    const statusEl = document.getElementById('status-' + rowId);
    const formData = new FormData();
    formData.append('file', file);

    // Animate bar while uploading
    let pct = 0;
    const interval = setInterval(() => {
      pct = Math.min(pct + Math.random() * 15, 85);
      bar.style.width = pct + '%';
    }, 200);

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const json = await res.json();
      clearInterval(interval);
      bar.style.width = '100%';
      if (json.ok) {
        statusEl.textContent = 'Done ✓';
        statusEl.className = 'file-status status-done';
        uploadedCount++;
        if (uploadedCount > 0) document.getElementById('ready-banner').style.display = 'flex';
      } else {
        statusEl.textContent = json.error || 'Error';
        statusEl.className = 'file-status status-error';
      }
    } catch(e) {
      clearInterval(interval);
      statusEl.textContent = 'Failed';
      statusEl.className = 'file-status status-error';
    }
  }
}
</script>
</body>
</html>
"""


class UploadHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet server logs
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/list":
            files = []
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                    size = f.stat().st_size
                    if size < 1024:
                        human = f"{size} B"
                    elif size < 1_048_576:
                        human = f"{size/1024:.1f} KB"
                    else:
                        human = f"{size/1_048_576:.1f} MB"
                    files.append({"name": f.name, "size": human})
            self.send_json({"files": files})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/upload":
            self.send_response(404)
            self.end_headers()
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"ok": False, "error": "Not multipart"}, status=400)
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            file_item = form["file"]
            filename = Path(file_item.filename).name
            ext = Path(filename).suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                self.send_json({"ok": False, "error": f"Format '{ext}' not allowed."}, status=400)
                return

            save_path = UPLOAD_DIR / filename
            with open(save_path, "wb") as f:
                f.write(file_item.file.read())

            print(f"  [UPLOAD] Saved: {filename} ({save_path.stat().st_size // 1024} KB)")
            self.send_json({"ok": True, "filename": filename})

        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    ip = get_local_ip()
    print("=" * 52)
    print("  🧠 LocalMind — Upload Server")
    print("=" * 52)
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{ip}:{PORT}")
    print()
    print("  Open the URL above in your browser.")
    print("  Drag & drop your files, then run main.py")
    print()
    print("  Allowed: .pdf  .txt  .csv  .docx")
    print("  Files will be saved to: data/raw/")
    print("=" * 52)
    print("  Press Ctrl+C to stop the server.")
    print()

    with socketserver.TCPServer(("0.0.0.0", PORT), UploadHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped. Run `python main.py` to start training.")
