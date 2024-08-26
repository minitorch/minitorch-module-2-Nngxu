from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """
        Set the mode of this module and all descendent modules to `train`.

        It can be implemented in a more concise way:
            ```
            self.training = True
            list = self.modules()
            for x in list:
                x.train()

            ```
        The following code is aimed to show structure of recursive function.
        """
        # base case
        if len(self.modules()) == 0:
            self.training = True
            return

        # recursive case
        self.training = True
        list = self.modules()
        for x in list:
            x.train()

    def eval(self) -> None:
        """
        Set the mode of this module and all descendent modules to `eval`(same as Inference).

        It can be implemented in a more concise way:
            ```
            self.training = False
            list = self.modules()
            for x in list:
                x.eval()

            ```

        The following code is aimed to show structure of recursive function.
        """
        # base case
        if len(self.modules()) == 0:
            self.training = False
            return

        # recursive case
        self.training = False
        list = self.modules()
        for x in list:
            x.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.

        It can be implemented in a more concise way:
            ```
            named_params = list(self._parameters.items())
            for name, mod in self._modules.items():
                named_params += [(name + "." + n, p) for n, p in mod.named_parameters()]
            return named_params
            ```

        Returns:
            The name and `Parameter` of each ancestor parameter.

        The following code is aimed to show structure of recursive function.
        """
        # base case
        # items() method returns a view object that can be iterated through as a tuple
        if len(self._modules.items()) == 0:
            named_params = list(self._parameters.items())
            return named_params

        # recursive case
        named_params = list(self._parameters.items())
        for name, mod in self._modules.items():
            named_params += [(name + "." + n, p) for n, p in mod.named_parameters()]
        return named_params

    def parameters(self) -> Sequence[Parameter]:
        """
        Enumerate over all the parameters of this module and its descendents.

        It can be implemented in a more concise way:
            ```
            allparam = list(self._parameters.values())
            for mod in self.modules():
                allparam += mod.parameters()
            return allparam
            ```

        The following code is aimed to show structure of recursive function.
        """
        if len(self._modules.items()) == 0:
            return list(self._parameters.values())

        allparam = list(self._parameters.values())
        for mod in self.modules():
            allparam += mod.parameters()
        return allparam

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        # __setattr__ will be call when we want to add a attribute for instance
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        "__getattr__ will be call when we want to get a unknown attribute for instance"
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        # base case
        if len(self._modules.items()) == 0:
            return self.__class__.__name__ + "()"

        # recursive case
        def _addindent(s_: str, numSpaces: int) -> str:
            """
            Add spaces to the beginning of each line of a string.

            Example:
                s = "Hello\nWorld\nThis is a test"
                print(_addindent(s, 4))

                Output:
                Hello
                    World
                    This is a test
            """
            # s = "Hello\nWorld\nThis is a test"
            # s.split("\n") -> output：['Hello', 'World', 'This is a test']
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            # The most important line of code in tree format output
            s2 = [(numSpaces * " ") + line for line in s2]
            # "_".join: using `_` concatenate between the elements of a list into a string
            # s = ["Hello", "World", "This", "is", "a", "test"], s1 = ["Hello"]
            # " ".join(s) -> output: 'Hello World This is a test'
            # " ".join(s1) -> output: 'Hello'
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            # Add two spaces at the begin of child modules
            # method: Add two spaces after all `\n`
            # because before the first `\n` is the parent module class name
            # cause by `main_str += "\n  " + "\n  ".join(lines) + "\n"`
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # add fomatting for same level modules
            # example:lines = '(c): ModuleA4()', main_str = ModuleA3(
            # output: "\n  ".join(lines) = '(c): ModuleA4()'
            #                   main_str = 'ModuleA3(\n  (c): ModuleA4()\n'
            # first "\n ": before "\n " -> parent module class name, here is `ModuleA3(`
            #              after "\n" -> chile module name + chile module Class name,here is `(c): ModuleA4()`
            # second "\n ": using `\n ` sepereate multiple child modules
            # final "\n": end of the string
            # Another example:
            # lines = ['(a): ModuleA2()', '(b): ModuleA3()']
            # main_str = 'ModuleA1('
            # after execute folllowing code:
            # main_str = 'ModuleA1(\n  (a): ModuleA2()\n  (b): ModuleA3()\n'
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
