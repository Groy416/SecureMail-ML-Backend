# Isolated synthetic SMTP/IMAP lab

This lab generates only SMTP or IMAP STARTTLS handshake traffic on an internal
Docker network. It has no published ports, accounts, mailbox access, external
recipients, or message bodies.

Run one scenario from the repository root:

```bash
uv run python -m datasets.lab.runner \
  --scenario normal_tls13_valid \
  --protocol SMTP \
  --environment lab_seed_0001 \
  --seed 420042
```

Use `--protocol IMAP` for an IMAP run. Each invocation writes
`datasets/lab/runs/<profile-hash>/` with:

- `runtime_profile.json`: the protocol-specific scenario manifest and derived seed.
- `capture.pcap`: packet capture for successful runs only.
- `run_manifest.json`: status, PCAP SHA-256, profile, platform, and local image IDs.
- `ca.pem` and `tls.keys`: lab-only extraction inputs; both are ignored by Git.

Run status is explicit:

- `success`: non-empty PCAP captured; it can be passed through `extract_run`.
- `unsupported_in_lab`: the requested cryptographic setting was refused by the local OpenSSL/TLS runtime. No PCAP feature row is emitted.
- `failed`: Docker, service, client, or capture failed. No PCAP feature row is emitted.

The current matrix handles normal TLS 1.3/1.2, STARTTLS-unused and aborted
negotiation behavior, certificate-chain/hostname profiles, RSA-without-forward-
secrecy, and repeated-session behavior. Legacy TLS, 3DES, weak RSA, and expired
certificate requests are retained only when the runtime can create them;
otherwise they are recorded as `unsupported_in_lab`.

A trainable `synthetic_pcap` dataset still needs successful records spanning
three isolated environments and all five risk labels. The current smoke run is
integration evidence, not a model-training dataset.
