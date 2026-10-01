"""Fail CI on known public OSV advisories in installed app/test dependencies."""
import importlib.metadata
import json
import urllib.request


def main():
    queries = [
        {"package": {"name": item.metadata["Name"], "ecosystem": "PyPI"}, "version": item.version}
        for item in importlib.metadata.distributions()
        if item.metadata["Name"].lower() not in {"pip", "setuptools"}
    ]
    request = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": queries}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        results = json.load(response)["results"]
    if len(results) != len(queries):
        raise RuntimeError("Advisory service returned an incomplete response")
    failures = 0
    for query, result in zip(queries, results):
        for vulnerability in result.get("vulns", []):
            failures += 1
            print(f"Known advisory: {query['package']['name']}=={query['version']} {vulnerability['id']}")
    if failures:
        raise SystemExit("Update affected dependencies before deploying")
    print(f"No known OSV advisories in {len(queries)} installed application/test packages (build tooling excluded)")


if __name__ == "__main__":
    main()
