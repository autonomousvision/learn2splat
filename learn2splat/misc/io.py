import pathlib
from copy import copy
import yaml
from colorama import Fore
from omegaconf import OmegaConf
from yaml.constructor import ConstructorError

KNOWN_TAGS = ["target", "context", "info", "debug"]


class CustomPath(pathlib.Path):
    """A custom path class that can be formatted to display as a hyperlink in terminal."""

    # This is a hack to inherit pathlib.Path and initialize the _flavour property.
    # https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
    # noinspection PyProtectedMember
    # noinspection PyUnresolvedReferences
    _flavour = type(pathlib.Path())._flavour

    def __format__(self, format_spec):
        if format_spec == '':
            return str(self)
        elif format_spec == 'link':
            if self.exists():
                return _create_hyperlink(self.resolve())
            else:
                # Missing path: find first existing parent
                missing_path = self.resolve()
                existing_parent = self.parent
                while existing_parent and not existing_parent.exists():
                    existing_parent = existing_parent.parent

                # Build base error message
                base_msg = f"\033[1;31m{missing_path} does not exist.\033[0m"

                if existing_parent and existing_parent.exists():
                    parent_link = _create_hyperlink(existing_parent.resolve())

                    # Gather existing parent’s contents
                    content_msg = ""
                    if existing_parent.is_dir():
                        content = list(existing_parent.iterdir())
                        if content:
                            content_msg = (
                                    "\n" + cyan("Nearest existing directory contents:") + "\n" +
                                    "\n".join(['  ' + _create_hyperlink(p.resolve()) for p in content])
                            )

                    return f"{base_msg}\nNearest existing directory: {parent_link}{content_msg}"
                else:
                    return f"{base_msg}\n(No existing parent found.)"
        elif format_spec.startswith('last'):
            i = int(format_spec[4:])
            return "/".join(self.parts[-i:])
        elif format_spec == 'exists':
            if self.exists():
                # Normal case: just print the link
                return _create_hyperlink(self.resolve())
            else:
                return _create_hyperlink(self.resolve()) + ' does not exist. \nParent directory: ' + _create_hyperlink(
                    self.parent.resolve())
        else:
            return str(self).__format__(format_spec)

    def __iadd__(self, other: str):
        return CustomPath(str(self) + other)

    def __add__(self, other: str):
        return CustomPath(str(self) + other)

    def __sub__(self, other):
        return CustomPath(self.resolve().relative_to(other.resolve()))


def _create_hyperlink(text: str | pathlib.Path):
    if isinstance(text, pathlib.Path):
        text = str(text)
    return f'file:///' + text.replace('\\', '/')


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class FrequencyScheduler:
    def __init__(
            self,
            last_step: int,
            frequencies: list[int] | None = None,
            steps: list[int] | None = None,
            iters: list[int] | None = None,
            enable_target: bool = True,
            enable_context: bool = True,
            enable_info: bool = True,
            enable_debug: bool = True,
    ):
        if iters is not None:
            print("FrequencyScheduler: using iters argument, ignoring frequencies and steps.")
            # assert frequencies is None and steps is None, "When iters is provided, frequencies and steps must be None"
        elif frequencies is None and steps is None:
            # Make sure frequencies and steps are both either None or lists of the same length
            frequencies = [99999999]  # effectively never
            steps = [0]
        elif frequencies is None or steps is None:
            raise ValueError("frequencies and steps must both be None or both be lists")
        else:
            assert len(frequencies) == len(
                steps), f"frequencies and steps must be same length. Got {len(frequencies)} and {len(steps)}"
            assert steps[0] == 0, f"first step must be 0. Got {steps}"

        if iters is not None:
            self.iterations = copy(iters)
            # check if last step in iters, else add it to iters and sort, remove higher than last_step
            self.iterations = sorted([i for i in self.iterations if i <= last_step])
            if last_step not in self.iterations:
                self.iterations.append(last_step)
            if 0 not in self.iterations:
                self.iterations.insert(0, 0)
        else:
            frequencies = copy(frequencies)
            steps = copy(steps)
            steps.pop(0)  # remove the first step which is always 0
            if last_step not in steps:
                steps.append(last_step)  # ensure last step is included
            pairs = list(zip(frequencies, steps))
            self.iterations: list[int] = self.get_all_iterations(pairs, last_step)

        self.verbose = False
        self.last_step = last_step

        self.enabled_tags = {
            "target": enable_target,
            "context": enable_context,
            "info": enable_info,
            "debug": enable_debug
        }

        self.is_disabled = False

    def set_all_tags(self, enabled: bool):
        for key in self.enabled_tags:
            self.enabled_tags[key] = enabled

    def check_iteration(self, iteration: int, tag: str) -> bool:
        """Returns True if any frequency event occurs at this iteration."""
        assert tag in KNOWN_TAGS, f"Invalid tag: {tag}, must be in {KNOWN_TAGS}"
        if self.enabled_tags[tag]:
            return iteration in self.iterations
        else:
            return False

    def _occurs_at(self, iteration: int, pairs, last_step) -> bool:
        """Returns True if any frequency event occurs at this iteration."""

        if iteration == last_step:
            return True

        for freq, end in pairs:
            if iteration <= end:
                if iteration % freq == 0:
                    return True
                else:
                    break

        return False

    def get_all_iterations(self, pairs, last_step) -> list[int]:
        """Returns a list of all iterations where an event occurs up to the last step."""
        t = 0
        iterations = []
        while t <= last_step:
            if self._occurs_at(t, pairs, last_step):
                iterations.append(t)
            t += 1
        return iterations

    def get_iterations(self, length_of_event: int) -> list[int]:
        """Iteration indices for the first `length_of_event` saved events.

        Callers pass len(render_list)/len(gaussian_list), whose position 0 is the init (iteration 0),
        so the labels are the first `length_of_event` scheduled iterations -- including for a partial
        scene that saved fewer than the full schedule.
        """
        if self.iterations is not None and len(self.iterations) >= length_of_event:
            return self.iterations[:length_of_event]
        else:
            raise ValueError(
                f"Not enough iterations up to last_step {self.last_step} to get {length_of_event} events. "
                f"Only got {len(self.iterations)} events.")

    def disable(self, flag):
        self.is_disabled = flag

    def __call__(self, iteration: int, tag: str = "") -> bool:
        if self.is_disabled:
            return False
        return self.check_iteration(iteration, tag)

    def __repr__(self):
        return f"FrequencyScheduler({self.iterations})"


def read_omega_cfg(path: pathlib.Path) -> OmegaConf:
    """Reads an OmegaConf YAML file, handling custom tags safely."""
    try:
        loaded_cfg = OmegaConf.load(path)
    except ConstructorError as e:
        # --- 1. Define a safe fallback constructor for the tag ---
        def custompath_constructor(loader, node):
            # Detect if it's a scalar or sequence
            if isinstance(node, yaml.ScalarNode):
                value = loader.construct_scalar(node)
                return CustomPath(value)
            elif isinstance(node, yaml.SequenceNode):
                seq = loader.construct_sequence(node)
                # joint the seq parts into a path
                path = CustomPath()
                for part in seq:
                    path = path / str(part)
                return path
            else:
                raise TypeError(f"Unsupported YAML node type for CustomPath: {type(node)}")

        # A saved config names the class with the module path of the package that wrote it,
        # which changed if the package was renamed, so match the
        # class by name under any module rather than listing the spellings.
        def custompath_multi_constructor(loader, suffix, node):
            if not suffix.endswith("misc.io.CustomPath"):
                raise ConstructorError(
                    None, None, f"unsupported python object in config: {suffix}", node.start_mark
                )
            return custompath_constructor(loader, node)

        yaml.add_multi_constructor(
            'tag:yaml.org,2002:python/object/apply:', custompath_multi_constructor
        )

        # --- 2. Load with PyYAML safely ---
        with open(path, "r") as f:
            raw_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # --- 3. Convert to OmegaConf ---
        loaded_cfg = OmegaConf.create(raw_cfg)

    return loaded_cfg


if __name__ == '__main__':

    print_every = FrequencyScheduler(
        frequencies=[1, 2, 5],
        steps=[0, 5, 10],
        last_step=37,
        # iters=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 56, 67, 100]    
    )
    for i in range(37 + 1):
        if print_every(i, "target"):
            pass

    print(print_every.get_iterations(15))
