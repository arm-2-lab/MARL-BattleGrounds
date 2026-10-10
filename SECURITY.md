# Security Policy

Report a security problem privately. Never post it in a public issue. Email
t.hawili@sms.ed.ac.uk, or use **Report a vulnerability** on the repository's
[Security tab](https://github.com/arm-2-lab/MARL-BattleGrounds/security)
when private vulnerability reporting is turned on there.

- The Replay Viewer and the BattleClient serve their pages on 127.0.0.1 only. In Docker,
  publish their ports on 127.0.0.1, as the install guide shows.
- Loading a method by `module:function` runs that Python code on your computer.
  Load only code you trust.
- Supported version: security fixes go into the latest release only.
