from vcr.request import Request as VcrRequest

def make_vcr_request(httpx_request, **kwargs):
    try:
        body = httpx_request.read().decode("utf-8")
    except Exception:
        body = httpx_request.read()
    uri = str(httpx_request.url)
    headers = dict(httpx_request.headers)
    return VcrRequest(httpx_request.method, uri, body, headers)
