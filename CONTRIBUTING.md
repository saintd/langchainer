# Contributing to LangChainer

LangChainer is an open-source project, and contributions are welcome!

## Reporting issues

If you encounter a bug or have a feature request, please [open an issue](https://github.com/saintd/langchainer/issues/new) on GitHub.

### Branching model

LangChainer uses the [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) branching model. This means that:

- The `main` branch is the "release branch" and always reflects the latest released version.
- The `develop` branch is the "development branch" where new features and bug fixes are developed.

When contributing code, please:

1. Fork the repository.
2. Create a feature branch from `develop`.
3. Make your changes.
4. Write unit tests and integration tests for your changes.
5. Build and test your changes.
6. Commit your changes with a descriptive commit message.
7. Open a pull request against the `develop` branch.

### Code style

Please follow the existing code style. The code should be formatted with `black` and checked with `flake8`.
