# Contributing to RAG-Based Recommendation System

First off, thank you for considering contributing to the RAG-Based Recommendation System project! We appreciate your interest and value your contributions. This document provides guidelines and best practices to help you contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Conventions](#coding-conventions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Styleguides](#styleguides)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Styleguide](#python-styleguide)
  - [Documentation Styleguide](#documentation-styleguide)
- [Attribution](#attribution)



## How to Contribute

### Reporting Issues

If you encounter any issues or bugs while using the RAG-Based Recommendation System, please [open an issue](https://github.com/ajliouat/rag-based-recommendation-system-with-phi-3-mini-4k-instruct/issues) on GitHub. When opening an issue, provide as much detail as possible, including:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots or error messages, if applicable
- System information (OS, Python version, etc.)

### Suggesting Enhancements

If you have ideas for new features or improvements, please [open an issue](https://github.com/ajliouat/rag-based-recommendation-system-with-phi-3-mini-4k-instruct/issues) on GitHub. Describe the enhancement in detail, including:

- The motivation behind the enhancement
- The proposed solution or implementation
- Any alternative solutions considered
- Potential benefits and drawbacks

### Pull Requests

We welcome pull requests to fix bugs, implement new features, or improve documentation. To submit a pull request:

1. Fork the repository and create a new branch from the `main` branch.
2. Make your changes, following the [coding conventions](#coding-conventions) and [styleguides](#styleguides).
3. Write appropriate tests to validate your changes.
4. Ensure that all tests pass and the code is free of linting errors.
5. Commit your changes with a clear and descriptive message.
6. Push your changes to your forked repository.
7. Open a pull request to the `main` branch of the original repository.
8. Provide a detailed description of your changes and the problem they solve.

We will review your pull request and provide feedback or merge it if it aligns with the project's goals and meets the necessary standards.

## Development Setup

To set up the development environment for the RAG-Based Recommendation System:

1. Fork the repository on GitHub.
2. Clone your forked repository:
   ```
   git clone git@github.com:ajliouat/rag-based-recommendation-system-with-phi-3-mini-4k-instruct.git
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up any necessary configuration files or environment variables.

## Coding Conventions

- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- Use meaningful variable and function names.
- Write docstrings for classes, methods, and functions.
- Keep functions and methods small and focused on a single responsibility.
- Use appropriate logging levels and messages.
- Handle errors and exceptions gracefully.

## Testing

- Write unit tests to cover critical functionality and edge cases.
- Use a testing framework such as `pytest` to organize and run tests.
- Ensure that all tests pass before submitting a pull request.
- Add new tests when introducing new features or fixing bugs.

## Documentation

- Keep the project's README and other documentation up to date.
- Document any new features, changes, or deprecations.
- Use clear and concise language in documentation.
- Provide examples and usage instructions where appropriate.

## Styleguides

### Git Commit Messages

- Use the present tense and imperative mood (e.g., "Add feature" instead of "Added feature").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.
- Consider starting the commit message with an applicable emoji:
  - 🎨 `:art:` when improving the format/structure of the code
  - 🐎 `:racehorse:` when improving performance
  - 🚱 `:non-potable_water:` when plugging memory leaks
  - 📝 `:memo:` when writing docs
  - 🐧 `:penguin:` when fixing something on Linux
  - 🍎 `:apple:` when fixing something on macOS
  - 🏁 `:checkered_flag:` when fixing something on Windows
  - 🐛 `:bug:` when fixing a bug
  - 🔥 `:fire:` when removing code or files
  - 💚 `:green_heart:` when fixing the CI build
  - ✅ `:white_check_mark:` when adding tests
  - 🔒 `:lock:` when dealing with security
  - ⬆️ `:arrow_up:` when upgrading dependencies
  - ⬇️ `:arrow_down:` when downgrading dependencies
  - 👕 `:shirt:` when removing linter warnings

### Python Styleguide

- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.
- Use 4 spaces for indentation (no tabs).
- Limit lines to 79 characters.
- Use snake_case for variable and function names.
- Use CamelCase for class names.
- Use docstrings to document modules, classes, and functions.
- Use type hints for function parameters and return values.

### Documentation Styleguide

- Use [Markdown](https://daringfireball.net/projects/markdown) for documentation.
- Use appropriate header levels (`#`, `##`, `###`, etc.) to structure the document.
- Use code blocks with language specifiers for code examples.
- Use italics for emphasis and bold for strong emphasis.
- Use lists and tables to present information clearly.

## Attribution

This CONTRIBUTING.md file is adapted from the [Atom Contributing Guidelines](https://github.com/atom/atom/blob/master/CONTRIBUTING.md).

