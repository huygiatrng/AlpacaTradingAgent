# Git branch naming

Never use `codex/` or `codex-` in branch names. The user requires plain, descriptive branch names without tool branding, such as `project-reliability-audit` or `fix-pr-48-macro-safety`. Follow this rule when creating or renaming branches, including when tools suggest a different default.

# Repository language

Keep source code, comments, documentation, tests, commit messages, and pull request descriptions in English. Do not add Vietnamese text to the repository.

# Runtime verification

Use any suitable available tools to test the running application. Browser testing is not restricted to IronBee DevTools or any other vendor.

# API checks

Keep reusable HTTP checks in `postman/collections/` and their usage in `postman/README.md`. Reuse existing collections instead of adding duplicate exports or generated vendor specifications. Never commit credentials, captured account data, or API run reports. Keep provider checks read-only; test order execution and failure handling with the isolated Python integration tests.
