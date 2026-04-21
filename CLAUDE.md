## Coding Philosophy
- Before writing any solution, propose 2-3 approaches ranked by simplicity
- Wait for approval before implementing
- Prefer built-ins and stdlib over custom logic
- No classes unless state genuinely needs to be managed
- Complexity must be explicitly justified — default to the simplest working solution
- Apply YAGNI: don't build for hypothetical future requirements
- Prefer flat over nested (Zen of Python)

## Code Review Habit
- After any implementation, self-critique for unnecessary complexity
- Ask: "What could be removed without losing functionality?"

## General Rules
- No scaffolding, boilerplate, or placeholder code unless explicitly asked
- If a built-in or 5-line solution exists, use it before reaching for a library
- One thing at a time — do it well before expanding scope
- Add `#` comment descriptions above every logical code block explaining what it does.
