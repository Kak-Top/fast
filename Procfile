web: gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:$PORT --timeout 120 --log-level info
