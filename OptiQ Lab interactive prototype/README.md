# OptiQ Lab interactive prototype

**This is a design prototype with sample data.** It is not the live OptiQ Lab server.

- Metrics (Capability Score, tok/s, port health) are **fictional**.
- Send / Export / Promote are **disabled**.
- For real Lab + Phase 0 spine, use `../mlx-optiq-dev` and `optiq lab`.

Open via a local static server (network required for React CDN unless vendored):

```bash
cd "OptiQ Lab interactive prototype"
python3 -m http.server 8765
# http://127.0.0.1:8765/OptiQ%20Lab.dc.html
```
