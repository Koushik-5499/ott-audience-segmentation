"""
Evaluator Service — OTT Audience Segmentation
===============================================
Responsibilities:
  1. Wait for API health (with timeout)
  2. Test GET /health
  3. Test POST /recommend with valid and edge-case profiles
  4. Collect clustering metrics from trainer output
  5. Generate comprehensive metrics.json
  6. Test cold-start / model-not-loaded scenario

All test results are reported honestly. Failures appear with real reasons.
"""

import os
import sys
import json
import time
import requests

API_URL = os.environ.get("API_URL", "http://api:8000")
COLD_API_URL = os.environ.get("COLD_API_URL", "http://api-cold:8000")
METRICS_PATH = os.environ.get("METRICS_PATH", "/app/output/metrics.json")
CLUSTERING_METRICS_PATH = os.environ.get(
    "CLUSTERING_METRICS_PATH", "/app/models/clustering_metrics.json"
)
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/models/metadata.json")
HEALTH_POLL_INTERVAL = 2
HEALTH_MAX_WAIT = 120


def wait_for_api_health(api_url: str, max_wait: int, interval: int) -> bool:
    """Poll GET /health until the API is responsive and model_loaded is True."""
    print(f"[INFO] Waiting for API at {api_url}/health (max {max_wait}s)...")
    elapsed = 0
    while elapsed < max_wait:
        try:
            resp = requests.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded") is True:
                    print(f"[INFO] API healthy and model loaded after {elapsed}s.")
                    return True
                else:
                    print("[INFO] API up but model not yet loaded. Waiting...")
            else:
                print(f"[WARN] /health returned status {resp.status_code}. Waiting...")
        except requests.exceptions.ConnectionError:
            print(f"[INFO] API not yet reachable ({elapsed}s elapsed). Waiting...")
        except Exception as e:
            print(f"[WARN] Unexpected error polling /health: {e}")
        time.sleep(interval)
        elapsed += interval

    print(f"[ERROR] API did not become healthy within {max_wait}s.")
    return False


def make_test_result(name, passed, expected=None, actual=None, error=None,
                     status_code=None, response_time_ms=None):
    """Create a standardized test result dict."""
    result = {
        "test": name,
        "passed": passed,
    }
    if expected is not None:
        result["expected"] = expected
    if actual is not None:
        result["actual"] = actual
    if error is not None:
        result["error"] = error
    if status_code is not None:
        result["status_code"] = status_code
    if response_time_ms is not None:
        result["response_time_ms"] = round(response_time_ms, 2)
    return result


def test_health(api_url: str) -> dict:
    """Test the GET /health endpoint."""
    try:
        start = time.time()
        resp = requests.get(f"{api_url}/health", timeout=5)
        elapsed_ms = (time.time() - start) * 1000

        data = resp.json()
        passed = (
            resp.status_code == 200
            and data.get("status") == "ok"
            and data.get("model_loaded") is True
        )
        return make_test_result(
            "GET /health",
            passed=passed,
            expected={"status": "ok", "model_loaded": True},
            actual=data,
            status_code=resp.status_code,
            response_time_ms=elapsed_ms,
        )
    except Exception as e:
        return make_test_result("GET /health", passed=False, error=str(e))


def test_recommend(api_url: str, payload: dict, test_name: str,
                   expect_status: int = 200, check_fn=None) -> dict:
    """Generic test for POST /recommend."""
    try:
        start = time.time()
        resp = requests.post(
            f"{api_url}/recommend",
            json=payload,
            timeout=10,
        )
        elapsed_ms = (time.time() - start) * 1000

        data = resp.json()

        if resp.status_code != expect_status:
            return make_test_result(
                test_name,
                passed=False,
                expected=f"HTTP {expect_status}",
                actual=f"HTTP {resp.status_code}",
                status_code=resp.status_code,
                response_time_ms=elapsed_ms,
            )

        if check_fn:
            check_passed, check_detail = check_fn(data, resp.status_code)
            return make_test_result(
                test_name,
                passed=check_passed,
                expected=f"HTTP {expect_status} + validation",
                actual=check_detail,
                status_code=resp.status_code,
                response_time_ms=elapsed_ms,
            )

        return make_test_result(
            test_name,
            passed=True,
            expected=f"HTTP {expect_status}",
            actual=f"HTTP {resp.status_code}",
            status_code=resp.status_code,
            response_time_ms=elapsed_ms,
        )

    except Exception as e:
        return make_test_result(test_name, passed=False, error=str(e))


def test_raw_request(api_url: str, body_str: str, test_name: str,
                     expect_status: int) -> dict:
    """Test with raw string body (for malformed JSON tests)."""
    try:
        start = time.time()
        resp = requests.post(
            f"{api_url}/recommend",
            data=body_str,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        elapsed_ms = (time.time() - start) * 1000

        passed = resp.status_code == expect_status
        try:
            data = resp.json()
        except Exception:
            data = resp.text

        # Check no stack traces exposed
        if isinstance(data, dict):
            detail = data.get("detail", "")
            if "traceback" in str(detail).lower() or "File \"" in str(detail):
                passed = False

        return make_test_result(
            test_name,
            passed=passed,
            expected=f"HTTP {expect_status}",
            actual=f"HTTP {resp.status_code}",
            status_code=resp.status_code,
            response_time_ms=elapsed_ms,
        )
    except Exception as e:
        return make_test_result(test_name, passed=False, error=str(e))


def validate_recommend_response(data, status_code):
    """Validate a successful /recommend response schema."""
    checks = []

    # Required fields
    for field in ["user_id", "segment_id", "segment_name", "recommendations", "distance_to_centroid"]:
        if field not in data:
            return False, f"Missing field: {field}"

    # Type checks
    if not isinstance(data["user_id"], str):
        return False, f"user_id should be str, got {type(data['user_id']).__name__}"
    if not isinstance(data["segment_id"], int):
        return False, f"segment_id should be int, got {type(data['segment_id']).__name__}"
    if not isinstance(data["segment_name"], str):
        return False, f"segment_name should be str, got {type(data['segment_name']).__name__}"
    if not isinstance(data["recommendations"], list):
        return False, f"recommendations should be list, got {type(data['recommendations']).__name__}"
    if not isinstance(data["distance_to_centroid"], (int, float)):
        return False, f"distance_to_centroid should be number, got {type(data['distance_to_centroid']).__name__}"

    # Value checks
    if data["segment_id"] < 0:
        return False, f"segment_id is negative: {data['segment_id']}"
    if data["distance_to_centroid"] < 0:
        return False, f"distance_to_centroid is negative: {data['distance_to_centroid']}"
    if not (0 <= data["distance_to_centroid"] < float("inf")):
        return False, f"distance_to_centroid not finite: {data['distance_to_centroid']}"
    if len(data["recommendations"]) == 0:
        return False, "recommendations list is empty"
    if not data["segment_name"]:
        return False, "segment_name is empty"

    return True, data


def test_cold_start(cold_api_url: str) -> list:
    """Test the cold-start scenario (API without model artifacts)."""
    results = []

    # Test 1: Health check should work but model_loaded=false
    try:
        resp = requests.get(f"{cold_api_url}/health", timeout=5)
        data = resp.json()
        passed = (
            resp.status_code == 200
            and data.get("status") == "ok"
            and data.get("model_loaded") is False
        )
        results.append(make_test_result(
            "Cold start: GET /health returns model_loaded=false",
            passed=passed,
            expected={"status": "ok", "model_loaded": False},
            actual=data,
            status_code=resp.status_code,
        ))
    except requests.exceptions.ConnectionError:
        results.append(make_test_result(
            "Cold start: GET /health returns model_loaded=false",
            passed=False,
            error=f"Could not connect to cold API at {cold_api_url}. "
                  "This is expected if api-cold service is not configured.",
        ))
    except Exception as e:
        results.append(make_test_result(
            "Cold start: GET /health returns model_loaded=false",
            passed=False, error=str(e),
        ))

    # Test 2: POST /recommend should return 503
    try:
        resp = requests.post(
            f"{cold_api_url}/recommend",
            json={
                "user_id": "test",
                "watch_time_hours": 10.0,
                "top_genres": ["Action"],
                "avg_session_mins": 30.0,
            },
            timeout=5,
        )
        data = resp.json()
        passed = resp.status_code == 503
        results.append(make_test_result(
            "Cold start: POST /recommend returns 503",
            passed=passed,
            expected="HTTP 503",
            actual=f"HTTP {resp.status_code}",
            status_code=resp.status_code,
        ))
    except requests.exceptions.ConnectionError:
        results.append(make_test_result(
            "Cold start: POST /recommend returns 503",
            passed=False,
            error=f"Could not connect to cold API at {cold_api_url}. "
                  "This is expected if api-cold service is not configured.",
        ))
    except Exception as e:
        results.append(make_test_result(
            "Cold start: POST /recommend returns 503",
            passed=False, error=str(e),
        ))

    return results


def load_clustering_metrics():
    """Load clustering metrics from trainer output."""
    try:
        if os.path.exists(CLUSTERING_METRICS_PATH):
            with open(CLUSTERING_METRICS_PATH, "r") as f:
                return json.load(f)
        else:
            print(f"[WARN] Clustering metrics not found at {CLUSTERING_METRICS_PATH}")
            return None
    except Exception as e:
        print(f"[WARN] Failed to load clustering metrics: {e}")
        return None


def load_metadata():
    """Load model metadata."""
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        else:
            return None
    except Exception:
        return None


def main():
    print("=" * 60)
    print("Evaluator Service Starting")
    print("=" * 60)

    all_tests = []
    latencies = []
    overall_status = "PASS"

    # --- Step 1: Wait for API ---
    healthy = wait_for_api_health(API_URL, HEALTH_MAX_WAIT, HEALTH_POLL_INTERVAL)
    if not healthy:
        overall_status = "FAIL"
        all_tests.append(make_test_result(
            "API health wait",
            passed=False,
            error=f"API did not become healthy within {HEALTH_MAX_WAIT}s",
        ))
        _write_metrics(build_output(all_tests, latencies, overall_status))
        sys.exit(1)

    # Load metadata for validation
    metadata = load_metadata()
    segment_names = metadata.get("segment_names", {}) if metadata else {}

    # --- Step 2: Test /health ---
    health_result = test_health(API_URL)
    all_tests.append(health_result)
    if health_result.get("response_time_ms"):
        latencies.append(health_result["response_time_ms"])
    print(f"[{'PASS' if health_result['passed'] else 'FAIL'}] {health_result['test']}")

    # --- Step 3: Valid representative profiles ---
    valid_profiles = [
        {
            "name": "High engagement action viewer",
            "payload": {
                "user_id": "USR-TEST-001",
                "watch_time_hours": 80.0,
                "top_genres": ["Action", "Thriller"],
                "avg_session_mins": 90.0,
                "sessions_per_week": 6.0,
                "weekend_ratio": 0.3,
            },
        },
        {
            "name": "Casual short-session viewer",
            "payload": {
                "user_id": "USR-TEST-002",
                "watch_time_hours": 5.0,
                "top_genres": ["Comedy"],
                "avg_session_mins": 15.0,
                "sessions_per_week": 1.0,
                "weekend_ratio": 0.7,
            },
        },
        {
            "name": "Genre explorer moderate viewer",
            "payload": {
                "user_id": "USR-TEST-003",
                "watch_time_hours": 35.0,
                "top_genres": ["Drama", "Sci-Fi", "Documentary", "Horror"],
                "avg_session_mins": 45.0,
                "sessions_per_week": 4.0,
                "weekend_ratio": 0.5,
            },
        },
        {
            "name": "Low activity weekend viewer",
            "payload": {
                "user_id": "USR-TEST-004",
                "watch_time_hours": 2.0,
                "top_genres": ["Family"],
                "avg_session_mins": 20.0,
                "sessions_per_week": 0.5,
                "weekend_ratio": 0.8,
            },
        },
    ]

    seen_segments = set()
    for vp in valid_profiles:
        result = test_recommend(
            API_URL,
            vp["payload"],
            f"Valid profile: {vp['name']}",
            expect_status=200,
            check_fn=validate_recommend_response,
        )
        all_tests.append(result)
        if result.get("response_time_ms"):
            latencies.append(result["response_time_ms"])
        if result["passed"] and isinstance(result.get("actual"), dict):
            seen_segments.add(result["actual"].get("segment_id"))
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # Check we hit multiple segments
    segment_coverage = make_test_result(
        "Segment coverage: diverse profiles hit multiple segments",
        passed=len(seen_segments) >= 2,
        expected="At least 2 different segments",
        actual=f"Hit {len(seen_segments)} segments: {sorted(seen_segments)}",
    )
    all_tests.append(segment_coverage)
    print(f"[{'PASS' if segment_coverage['passed'] else 'FAIL'}] {segment_coverage['test']}")

    # --- Step 4: Segment name validation ---
    if metadata and segment_names:
        for vp in valid_profiles:
            try:
                resp = requests.post(f"{API_URL}/recommend", json=vp["payload"], timeout=10)
                data = resp.json()
                if resp.status_code == 200:
                    sid = str(data.get("segment_id", ""))
                    expected_name = segment_names.get(sid, None)
                    actual_name = data.get("segment_name", "")
                    passed = expected_name is not None and expected_name == actual_name
                    result = make_test_result(
                        f"Segment name matches metadata for {vp['payload']['user_id']}",
                        passed=passed,
                        expected=expected_name,
                        actual=actual_name,
                    )
                    all_tests.append(result)
                    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")
            except Exception:
                pass

    # --- Step 5: Edge cases ---

    # 5a: Unseen genre
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-001",
            "watch_time_hours": 20.0,
            "top_genres": ["AlienGenre", "NonExistent"],
            "avg_session_mins": 30.0,
        },
        "Edge: unseen genre values",
        expect_status=200,
        check_fn=validate_recommend_response,
    )
    all_tests.append(result)
    if result.get("response_time_ms"):
        latencies.append(result["response_time_ms"])
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5b: Empty top_genres
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-002",
            "watch_time_hours": 20.0,
            "top_genres": [],
            "avg_session_mins": 30.0,
        },
        "Edge: empty top_genres list",
        expect_status=200,
        check_fn=validate_recommend_response,
    )
    all_tests.append(result)
    if result.get("response_time_ms"):
        latencies.append(result["response_time_ms"])
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5c: Zero watch time
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-003",
            "watch_time_hours": 0.0,
            "top_genres": ["Comedy"],
            "avg_session_mins": 10.0,
        },
        "Edge: zero watch time",
        expect_status=200,
        check_fn=validate_recommend_response,
    )
    all_tests.append(result)
    if result.get("response_time_ms"):
        latencies.append(result["response_time_ms"])
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5d: Very large watch time
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-004",
            "watch_time_hours": 999.0,
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: very large watch time (999)",
        expect_status=200,
        check_fn=validate_recommend_response,
    )
    all_tests.append(result)
    if result.get("response_time_ms"):
        latencies.append(result["response_time_ms"])
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5e: Very large session duration
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-005",
            "watch_time_hours": 50.0,
            "top_genres": ["Drama"],
            "avg_session_mins": 999.0,
        },
        "Edge: very large session duration (999)",
        expect_status=200,
        check_fn=validate_recommend_response,
    )
    all_tests.append(result)
    if result.get("response_time_ms"):
        latencies.append(result["response_time_ms"])
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5f: Missing required field (user_id)
    result = test_recommend(
        API_URL,
        {
            "watch_time_hours": 20.0,
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: missing required field (user_id)",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5g: Missing required field (watch_time_hours)
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-006",
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: missing required field (watch_time_hours)",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5h: String where number expected
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-007",
            "watch_time_hours": "twenty",
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: string where number expected",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5i: Negative watch time
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-008",
            "watch_time_hours": -5.0,
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: negative watch_time_hours",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5j: Negative session mins
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-009",
            "watch_time_hours": 10.0,
            "top_genres": ["Action"],
            "avg_session_mins": -10.0,
        },
        "Edge: negative avg_session_mins",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5k: Malformed JSON
    result = test_raw_request(
        API_URL,
        "{bad json!!!",
        "Edge: malformed JSON body",
        expect_status=400,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5l: Repeated identical requests (idempotency)
    repeat_payload = {
        "user_id": "USR-REPEAT-001",
        "watch_time_hours": 40.0,
        "top_genres": ["Drama", "Comedy"],
        "avg_session_mins": 50.0,
        "sessions_per_week": 3.0,
        "weekend_ratio": 0.4,
    }
    responses = []
    for i in range(3):
        try:
            resp = requests.post(f"{API_URL}/recommend", json=repeat_payload, timeout=10)
            responses.append(resp.json())
        except Exception as e:
            responses.append({"error": str(e)})

    if len(responses) == 3:
        identical = (
            responses[0] == responses[1] == responses[2]
            and responses[0].get("segment_id") is not None
        )
    else:
        identical = False

    result = make_test_result(
        "Edge: repeated identical requests produce identical responses",
        passed=identical,
        expected="All 3 responses identical",
        actual=f"Identical={identical}, responses_count={len(responses)}",
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # 5m: Absurdly large watch_time (above max)
    result = test_recommend(
        API_URL,
        {
            "user_id": "USR-EDGE-010",
            "watch_time_hours": 5000.0,
            "top_genres": ["Action"],
            "avg_session_mins": 30.0,
        },
        "Edge: absurdly large watch_time (5000, above max 1000)",
        expect_status=422,
    )
    all_tests.append(result)
    print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['test']}")

    # --- Step 6: Cold start tests ---
    print("\n[INFO] Testing cold-start scenario...")
    cold_results = test_cold_start(COLD_API_URL)
    all_tests.extend(cold_results)
    for cr in cold_results:
        print(f"[{'PASS' if cr['passed'] else 'FAIL'}] {cr['test']}")

    # --- Step 7: Load clustering metrics ---
    clustering_metrics = load_clustering_metrics()

    # --- Build final output ---
    output = build_output(all_tests, latencies, overall_status, clustering_metrics, metadata)
    _write_metrics(output)

    # Print summary
    passed = sum(1 for t in all_tests if t.get("passed"))
    total = len(all_tests)
    failed = total - passed
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {passed}/{total} tests passed, {failed} failed")
    print(f"metrics.json written to: {METRICS_PATH}")
    print(f"{'=' * 60}")

    # Exit non-zero if any test failed
    if failed > 0:
        sys.exit(1)
    sys.exit(0)


def build_output(all_tests, latencies, overall_status, clustering_metrics=None, metadata=None):
    """Build the final metrics.json output."""
    passed = sum(1 for t in all_tests if t.get("passed"))
    total = len(all_tests)
    failed = total - passed

    if failed > 0:
        overall_status = "FAIL"

    # Latency stats
    latency_stats = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_stats = {
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "p95_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 2) if len(latencies_sorted) > 1 else round(latencies_sorted[0], 2),
            "max_ms": round(max(latencies), 2),
            "min_ms": round(min(latencies), 2),
            "count": len(latencies),
        }

    output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_url": API_URL,
        "overall_status": overall_status,
        "summary": {
            "tests_passed": passed,
            "tests_total": total,
            "tests_failed": failed,
        },
        "latency": latency_stats,
        "tests": all_tests,
    }

    # Include clustering metrics if available
    if clustering_metrics:
        output["clustering_metrics"] = clustering_metrics

    # Include segment metadata
    if metadata:
        output["model_metadata"] = {
            "n_clusters": metadata.get("n_clusters"),
            "segment_names": metadata.get("segment_names"),
            "segment_profiles": metadata.get("segment_profiles"),
            "is_synthetic_data": metadata.get("is_synthetic_data", False),
            "sklearn_version": metadata.get("sklearn_version"),
            "seed": metadata.get("seed"),
        }

    output["note"] = (
        "This evaluation was run on SYNTHETIC data. "
        "Silhouette and stability results reflect the generator, not real viewer behavior."
    )

    return output


def _write_metrics(data: dict):
    """Write results to metrics.json atomically."""
    try:
        out_dir = os.path.dirname(METRICS_PATH)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        tmp_path = METRICS_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, METRICS_PATH)
        print(f"[INFO] metrics.json written to {METRICS_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to write metrics.json: {e}")
        # Try writing directly as fallback
        try:
            with open(METRICS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e2:
            print(f"[ERROR] Fallback write also failed: {e2}")


if __name__ == "__main__":
    main()
