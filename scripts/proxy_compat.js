const http = require('http');

function createProxy(listenPort, targetPort, name) {
  const server = http.createServer((req, res) => {
    const options = {
      hostname: '127.0.0.1',
      port: targetPort,
      path: req.url,
      method: req.method,
      headers: { ...req.headers, host: `127.0.0.1:${targetPort}` },
    };
    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });
    proxyReq.on('error', (err) => {
      res.writeHead(502, { 'Content-Type': 'text/plain' });
      res.end(`Proxy error forwarding to port ${targetPort}: ${err.message}`);
    });
    req.pipe(proxyReq, { end: true });
  });

  server.listen(listenPort, '0.0.0.0', () => {
    console.log(`[Proxy] ${name}: Forwarding http://localhost:${listenPort} -> http://127.0.0.1:${targetPort}`);
  });
  return server;
}

createProxy(3007, 3000, 'Frontend (3007 -> 3000)');
createProxy(8007, 8000, 'Backend (8007 -> 8000)');

