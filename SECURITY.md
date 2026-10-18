# Security policy

## Reporting a vulnerability

Please report security problems **privately** through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue for them.

Include the nullcheck version, a minimal synthetic reproduction and what an attacker could achieve. This is a small volunteer project: there is no bug bounty, and responses are best effort.

Only the latest release receives fixes.

## Scope

nullcheck runs your feature code and strategies with your permissions, and `replay` imports and runs the code a record refers to. **Running or replaying untrusted code is unsafe by design and is documented, not a vulnerability.**

In scope, for example:
- nullcheck writing outside the `--out` folder, or overwriting an existing folder;
- `replay` or `validate` following a path outside `--root`, or running code despite a refused or changed hash without `--allow-modified`;
- a crafted record that passes validation but makes nullcheck itself misbehave;
- nullcheck sending data anywhere.
