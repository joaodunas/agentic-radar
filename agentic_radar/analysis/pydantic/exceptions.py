class InvalidPydanticAIAgentConstructorError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class InvalidToolDecoratorError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class InvalidSystemPromptDecoratorError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class InvalidDependencyTypeError(Exception):
    def __init__(self, *args):
        super().__init__(*args) 