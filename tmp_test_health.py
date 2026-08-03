import json
import urllib.request

data = json.dumps({'features':[35,'Male',78,98,37.2]}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/health', data=data, headers={'Content-Type':'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(r.read().decode())
except Exception as e:
    print(type(e).__name__, e)
