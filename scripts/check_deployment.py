"""Small, unauthenticated availability probe; never submits data or spends tokens."""
import urllib.request
import http.cookiejar

URL = "https://groq-data-analyst.streamlit.app/~/+/_stcore/health"


def main():
    request = urllib.request.Request(URL, headers={"User-Agent": "AI-Analyst-Healthcheck/1"})
    # Community Cloud redirects may establish a routing cookie first.
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(request, timeout=20) as response:
        body = response.read(1024).decode("utf-8").strip().lower()
        if response.status != 200 or body != "ok":
            raise RuntimeError("Deployment health check failed")
    print("Deployment health: OK (server availability only; authenticated analysis requires a separate smoke test)")


if __name__ == "__main__":
    main()
