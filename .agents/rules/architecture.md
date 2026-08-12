# Architecture Rules

This file contains project-specific architectural constraints.

## Rule Priority

When implementation choices conflict, use this order:

1. explicit current user requirement
2. current repository architecture and tests
3. this file
4. project documentation
5. historical assumptions

## Project Architecture

Document stable architecture here as the project evolves.

For every important boundary, record:

- responsibility
- allowed dependencies
- forbidden dependencies
- authority/ownership rules
- persistence boundaries
- I/O boundaries
- test expectations

Do not add speculative architecture merely because it may be useful later.
