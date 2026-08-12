# Project Agent Instructions

## Authority

The main Gemini agent is the Lead/Integrator for this repository.

The Lead owns:

- interpretation of user intent
- project scope
- architectural decisions
- task decomposition
- subagent delegation
- integration
- final completion decisions

Subagents perform bounded delegated work. They must not silently expand their scope.

## Repository First

Before modifying repository behavior:

1. Inspect the current repository state.
2. Read relevant project and architecture documentation.
3. Inspect existing tests and implementation patterns.
4. Run appropriate baseline verification when practical.
5. Reconcile discrepancies between documentation and repository evidence.

Current repository evidence outranks assumptions in prompts or stale handoff documents.

## Superpowers

Use relevant Superpowers skills before acting.

For new features, behavioral changes, or design work:

1. use brainstorming
2. obtain design approval
3. use writing-plans
4. execute the approved plan

For bugs:

1. use systematic-debugging
2. establish the root cause
3. use test-driven-development where applicable
4. implement the smallest correct fix

Before claiming completion, use verification-before-completion.

When executing a multi-task implementation plan, prefer subagent-driven-development.

## Mandatory Delegation Pipeline

All non-trivial repository-changing work must use:

Architect → Builder → Reviewer → Verifier → Lead

The Lead may not skip a stage merely because the implementation appears easy.

Exceptions:

- explanation-only requests
- repository navigation
- status questions
- trivial typo or documentation corrections with no behavioral effect

If uncertain whether a change is trivial, treat it as non-trivial.

## Agent Responsibilities

### Architect

Investigates the repository and proposes a bounded implementation approach.

Architect normally does not modify production code.

### Builder

Implements the accepted task.

Builder must stay within scope, preserve architecture, add or update tests when warranted, and report exactly what verification was run.

Builder does not declare the milestone complete.

### Reviewer

Performs independent adversarial review.

Reviewer looks for:

- correctness bugs
- regressions
- architecture violations
- scope creep
- missing tests
- invalid assumptions
- unsafe shortcuts
- weakened tests
- incomplete error handling
- unnecessary complexity

Reviewer normally does not modify production code.

### Verifier

Independently verifies the integrated result.

Verifier runs the required build, tests, linting, type checks, architecture checks, and other project-specific completion commands.

Verifier does not repair failures.

## Test Integrity

Never:

- delete a legitimate failing test merely to obtain green
- weaken an assertion merely to obtain green
- skip a relevant test merely to obtain green
- make a failing path unreachable merely to obtain green
- change expected behavior without explicit architectural justification

Tests defend behavior and architecture; they are not obstacles to implementation.

## Scope Discipline

Prefer the smallest coherent change that satisfies the approved design.

Do not perform unrelated refactoring.

Do not introduce new dependencies, authority, infrastructure, abstractions, or behavior unless required by the approved scope.

## Completion

A change is complete only when:

- repository state was inspected
- relevant architecture was considered
- implementation matches approved scope
- tests were added or updated where warranted
- Builder verification passed
- Reviewer has no unresolved blocking findings
- Verifier independently reproduced required checks
- no tests were weakened merely to obtain green
- relevant documentation was updated
- the Lead reviewed the completion evidence

Only the Lead may declare the overall task or milestone complete.
