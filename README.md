# Alignment Tax Analysis

项目总览与运行说明见 [docs/README.md](/Users/shiqi/code/graduation-project/docs/README.md)。

## AI Collaboration Rule

This repository has a macOS-specific execution constraint:

- Any command or code path that depends on `MPS` must **not** be executed inside Codex.
- AI collaborators should prepare the exact terminal command for the user to run in the normal macOS Terminal.
- The user runs the command locally and sends the output back.
- Commands and code paths that do **not** depend on `MPS` may be executed by AI collaborators inside Codex as usual.

Examples of tasks that should be run by the user in Terminal instead of Codex:

- local model runs that require `device="mps"`
- PyTorch `MPS` availability checks
- local white-box probe / hidden-state extraction on Apple GPU
- any validation whose success depends on `torch.backends.mps.is_available()`

Examples of tasks that may still be run inside Codex:

- CPU fallback runs
- code edits
- tests unrelated to `MPS`
- data processing, metrics rebuilds, and report generation
- log inspection and integration checks

If an AI collaborator is unsure whether a task touches `MPS`, default to:

1. do not run it inside Codex
2. write a copy-paste terminal command for the user
3. wait for the user's terminal output before proceeding

## Multi-Department Workflow Rule

This workspace is jointly maintained under a multi-department AI workflow.
The canonical full rule lives in [docs/README.md](/Users/shiqi/code/graduation-project/docs/README.md).

Always remember:

- The workspace is handled by seven roles: `Architecture`, `Code`, `Analysis`, `Innovation`, `Documentation`, `Paper Writing`, and the `User` as dispatch hub.
- `Paper Writing` only handles paper-writing tasks such as main-text drafting, appendix wording, narrative structure, and submission-language polishing. Work already owned by the other five AI departments should not be routed to `Paper Writing`.
- AI departments must **not** communicate directly with each other.
- All workflow handoff must go through the user.
- Every department response should stay structured and concise.
- Every department response must include a forwarding prompt with both source and target departments.

Default handoff pattern:

1. User -> Department A
2. Department A -> structured output
3. User forwards to Department B

Forwarding prompts must use this pattern:

```text
【FROM: Department A】
【TO: Department B】
```
