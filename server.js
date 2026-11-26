// CAD 三维搜索项目 - 项目全景网站（带在线认领分工）
// 零依赖 Node HTTP 服务：静态托管 + /api/claims 认领数据
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');
const DATA_FILE = path.join(DATA_DIR, 'claims.json');

// 认领数据：roleIndex -> 姓名。优先内存，其次落盘（落盘失败不影响使用）
let claims = {};
try {
  claims = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  if (typeof claims !== 'object' || claims === null) claims = {};
} catch (e) { claims = {}; }

function saveClaims() {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(DATA_FILE, JSON.stringify(claims, null, 2));
  } catch (e) { /* 只读环境：降级为内存存储 */ }
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.doc': 'application/msword',
  '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  '.ppt': 'application/vnd.ms-powerpoint',
  '.md': 'text/markdown; charset=utf-8',
  '.py': 'text/plain; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.stp': 'application/octet-stream',
  '.pdf': 'application/pdf'
};

function serveStatic(req, res, urlPath) {
  let p;
  try {
    // URL 中的中文/特殊字符是百分号编码的，必须先解码再找文件
    p = decodeURIComponent(urlPath);
  } catch (e) {
    p = urlPath;
  }
  p = p === '/' ? '/index.html' : p;
  p = p.split('?')[0].split('#')[0].replace(/\//g, path.sep);
  p = path.normalize(p).replace(/^(\.\.[\/\\])+/, '');
  const file = path.join(ROOT, p);
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end('Forbidden'); }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('Not Found'); }
    const ext = path.extname(file).toLowerCase();
    const headers = { 'Content-Type': MIME[ext] || 'application/octet-stream' };
    // Office 文件提示下载（带中文文件名），文本类文件直接在浏览器打开
    if (['.docx', '.doc', '.pptx', '.ppt'].includes(ext)) {
      headers['Content-Disposition'] = 'attachment; filename*=UTF-8\'\'' + encodeURIComponent(path.basename(file));
    }
    res.writeHead(200, headers);
    res.end(buf);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');

  if (url.pathname === '/api/claims' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    return res.end(JSON.stringify(claims));
  }

  if (url.pathname === '/api/claims' && req.method === 'POST') {
    let body = '';
    req.on('data', c => { body += c; if (body.length > 10000) req.destroy(); });
    req.on('end', () => {
      try {
        const d = JSON.parse(body);
        const role = String(parseInt(d.role, 10));
        const name = String(d.name || '').trim().slice(0, 20);
        const action = d.action || 'claim';
        if (!/^\d+$/.test(role) || parseInt(role, 10) < 0 || parseInt(role, 10) > 9) {
          res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
          return res.end(JSON.stringify({ error: 'bad role' }));
        }
        if (action === 'release') {
          delete claims[role];
        } else if (name) {
          // 一个名字只能认领一个分工：先清掉该名字在其他分工上的认领
          for (const k of Object.keys(claims)) {
            if (claims[k] === name && k !== role) delete claims[k];
          }
          claims[role] = name;
        }
        saveClaims();
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
        res.end(JSON.stringify(claims));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: 'bad request' }));
      }
    });
    return;
  }

  serveStatic(req, res, url.pathname);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log('CAD project site running on port ' + PORT);
});
