# AlphaEdge API

NSE Trading Signal API — Flask Backend

## Endpoints

- GET /api/health — Check API is running
- GET /api/market — Get Nifty market context
- GET /api/scan?n=10&min_score=75 — Scan top N stocks
- GET /api/stock/HDFCBANK — Single stock signal

## Deploy on Render.com (Free)

1. Push this folder to GitHub
2. Go to render.com
3. New Web Service → connect GitHub repo
4. Build command: pip install -r requirements.txt
5. Start command: gunicorn app:app
6. Deploy — get free URL
