# Transactional inbox: local API implementation study

Original, uncommissioned implementation demonstration prepared for a Cyflogic pitch. Synthetic event IDs and integer balances; no client data, financial transaction, production deployment or professional operations history is represented.

## Run

Python standard library only; no installation, network, account or API cost.

```
python -m unittest -v test_inbox.py
```

Run from this directory. Seven tests cover a first event, replay after opening another connection, changed-payload conflict, two injected transaction failures, eight concurrent duplicate attempts with separate connections, and invalid input rejection. Temporary databases are cleaned up by the tests.

## What it proves

For one SQLite database, a unique event key, `BEGIN IMMEDIATE`, and a single transaction containing the inbox row and business update allow the tested duplicate attempts to produce one committed balance change. Raising before commit rolls both changes back. A reused key with different canonical payload is rejected.

## Limits

This is the persistence core, not a deployed webhook endpoint. It does not implement HTTP handling, signature/authentication checks, HTTP acknowledgement timing, distributed consumers, external payment/email effects, expiry, multi-tenant key scope, rate limits, production migrations, disk-loss recovery, or process-kill/power-loss fault injection. SQLite serializes writers; the test is not a throughput benchmark. Real workloads need a retention/idempotency horizon, provider retry contract, tenant-scoped keys and signature validation before processing. External effects require their own idempotency contract or an outbox workflow; this transaction cannot make a remote API atomic. No general exactly-once guarantee is claimed.

The article pitch would use current primary SQLite transaction and Python sqlite3 documentation, with these boundaries explicit. No complete unpaid article is included.


## Source: inbox.py

```python
"""Local transactional inbox demonstration; no HTTP/security/production claim."""
import hashlib
import json
import sqlite3

class PayloadConflict(ValueError):
    pass

def connect(path):
    con = sqlite3.connect(path, timeout=5, isolation_level=None)
    con.execute("CREATE TABLE IF NOT EXISTS inbox(event_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY CHECK(id=1), balance INTEGER NOT NULL)")
    con.execute("INSERT OR IGNORE INTO account VALUES(1,0)")
    return con

def process(con, event_id, payload, fail_at=None):
    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128:
        raise ValueError("invalid event identifier")
    if not isinstance(payload, dict) or set(payload) != {"amount"}:
        raise ValueError("expected one amount field")
    amount = payload["amount"]
    if type(amount) is not int or not 0 < amount <= 1000000:
        raise ValueError("amount must be a bounded positive integer")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    con.execute("BEGIN IMMEDIATE")
    try:
        prior = con.execute("SELECT digest FROM inbox WHERE event_id=?", (event_id,)).fetchone()
        if prior:
            if prior[0] != digest:
                raise PayloadConflict("event identifier reused with a different payload")
            con.execute("COMMIT")
            return "duplicate"
        con.execute("INSERT INTO inbox VALUES(?,?)", (event_id, digest))
        if fail_at == "before_effect":
            raise RuntimeError("injected failure")
        con.execute("UPDATE account SET balance=balance+? WHERE id=1", (amount,))
        if fail_at == "after_effect":
            raise RuntimeError("injected failure")
        con.execute("COMMIT")
        return "applied"
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise

```

## Tests: test_inbox.py

```python
import concurrent.futures
from pathlib import Path
import tempfile
import threading
import unittest
from inbox import PayloadConflict, connect, process

class InboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "demo.db")
        self.con = connect(self.path)
    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()
    def state(self):
        return (self.con.execute("SELECT balance FROM account").fetchone()[0], self.con.execute("SELECT count(*) FROM inbox").fetchone()[0])
    def test_one_event(self):
        self.assertEqual(process(self.con, "a", {"amount": 5}), "applied")
        self.assertEqual(self.state(), (5, 1))
    def test_duplicate_after_reopen(self):
        process(self.con, "a", {"amount": 5})
        other = connect(self.path)
        try:
            self.assertEqual(process(other, "a", {"amount": 5}), "duplicate")
        finally:
            other.close()
        self.assertEqual(self.state(), (5, 1))
    def test_payload_conflict(self):
        process(self.con, "a", {"amount": 5})
        with self.assertRaises(PayloadConflict):
            process(self.con, "a", {"amount": 9})
        self.assertEqual(self.state(), (5, 1))
    def test_failure_before_effect_can_retry(self):
        with self.assertRaises(RuntimeError):
            process(self.con, "a", {"amount": 5}, "before_effect")
        self.assertEqual(self.state(), (0, 0))
        self.assertEqual(process(self.con, "a", {"amount": 5}), "applied")
        self.assertEqual(self.state(), (5, 1))
    def test_failure_after_effect_can_retry(self):
        with self.assertRaises(RuntimeError):
            process(self.con, "a", {"amount": 5}, "after_effect")
        self.assertEqual(self.state(), (0, 0))
        process(self.con, "a", {"amount": 5})
        self.assertEqual(self.state(), (5, 1))
    def test_eight_concurrent_duplicate_attempts(self):
        barrier = threading.Barrier(8)
        def worker(_):
            con = connect(self.path)
            try:
                barrier.wait(timeout=5)
                return process(con, "same", {"amount": 7})
            finally:
                con.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(8)))
        self.assertEqual(results.count("applied"), 1)
        self.assertEqual(results.count("duplicate"), 7)
        self.assertEqual(self.state(), (7, 1))
    def test_invalid_input_has_no_effect(self):
        for payload in [{"amount": True}, {"amount": 0}, {"amount": "5"}, {"amount": 1000001}, {"amount": 2, "extra": 1}]:
            with self.assertRaises(ValueError):
                process(self.con, "a", payload)
        self.assertEqual(self.state(), (0, 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)

```

## Producer test result

```json
{
  "observed_at": "2026-10-19T15:48:39.900425+00:00",
  "command": "python -m unittest -v test_inbox.py",
  "cwd": "C:\\Users\\sgife\\dev\\sorinOS\\runtime\\evidence\\paid-outreach-20261019-continue\\design\\cyflogic\\fixture",
  "python": "3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]",
  "sqlite": "3.50.4",
  "observed_exit_code": 0,
  "observed_tests": 7,
  "observed_duration_seconds": 0.789,
  "outcome": "PASS_PRODUCER_RUN",
  "coverage": [
    "first event",
    "duplicate from reopened separate connection",
    "changed payload conflict",
    "rollback before business effect",
    "rollback after business effect before commit",
    "eight separate connection concurrent duplicate attempts:1applied7duplicate",
    "invalidpayload rejection"
  ],
  "limits": [
    "No HTTP/signature verification",
    "No processkill orpowerlossfaultinjection",
    "No externalAPI exactlyonceclaim",
    "No throughputbenchmark",
    "No productionorclienthistory evidence"
  ],
  "prior_runner_error": "Initial unittest invoked fromrepo root produced ModuleNotFoundError; corrected workingdirectory only; fixture tests thenpassed. No hidden code repair."
}
```

# Sources and provenance

Original implementation and tests, prepared locally on 19 October 2026 for a proposed paid editorial pitch. No customer code or data. No third-party code was copied.

Primary references inspected:
- https://www.sqlite.org/lang_transaction.html — manual transactions, single writer, BEGIN IMMEDIATE and SQLITE_BUSY.
- https://docs.python.org/3/library/sqlite3.html — explicit transaction control, parameter binding, connections and thread limits. The current /3/ page identifies Python 3.15; observed fixture runtime is recorded separately. No claim of testing that documentation runtime.

The local model uses Python standard-library modules only. Existing Python/SQLite licences govern their runtimes; no runtime is redistributed in the archive.
