# Daily preparation source

This is a local, unpublished preparation tool recovered from branch
`6b71f629a6ad6204b2822984825ad462818ee948` with security repairs.

All source entries are disabled. Historical rights/speaker/editorial claims are
unverified. The configured local media and cover renderer are absent; rendering
stops explicitly until a reviewed implementation and actual inputs are supplied.
No scheduler, upload helper, publication, paid route or daemon is installed.
Ollama is optional via `--local-llm` and must already be running.

The Jev gate never loads a backend without an explicit `sourceEvidence` receipt
bound to the exact outgoing normalized opening, transcript and description.
Required fields: `status=VERIFIED_PUBLIC`, HTTPS NASA `sourceUrl`, `reviewedAt`
(timezone-aware), `reviewedBy`, and `contentSha256` from `payload_sha256(item)`.
This validates a supplied review receipt; it does not independently verify rights.
Missing, failed, partial or malformed evaluation remains `UNVERIFIED`. A scored
text result is not visual/audio approval, publishing approval or income evidence.

`picks/2026-09-27.json` is a historical unpublished editorial draft. It is not a
receipt that those clips were rendered, accepted, published or earned revenue.

Offline regression checks:
`python -m unittest discover -s clipper/daily -p "test_*.py"`.
These test only the gate and pure path/provenance guards, not media rendering.
