# Contributing to TorchLego

## TL;DR
- We do not accept Issues/Pull Requests not suitable to the project e.g.
  "Can TorchLego support Tensorflow models or TFLite stuff?", the answer is straight NO.
- Create GitHub Issues or Pull Requests to ask/give feedbacks, label the issue/pull request
  and assign them to the maintainer of the repository for any communication.
- If you are contributing Pull Requests, ensure that code is linted using `pylint` and formatted properly along with some unit tests for your change.

## Development

### Installation
- For development you have following prerequisities:
  - [Docker](https://www.docker.com/)
  - [Python 3.10](https://www.python.org/)
  - [Understanding YAML](https://yaml.org/)
- You can install the requirements for the development of TorchLego using:
```
pip install -r requirements.txt
```
- You can install following packages for linting & testing:

```
pip install pylint pytest
```

### Unit Testing
- The unit tests are added in `tests` directory for majority of the functionality.
- Ensure `tests` directory has subdirectories as per the functionality packages e.g. `config`, `api`, `core`, etc.
- Unit tests should cover both positive/negative case scenarios.

### Documentation
TBD

## Examples
- Submit Pull Requests with custom examples of models and transforms to `examples` directory.
