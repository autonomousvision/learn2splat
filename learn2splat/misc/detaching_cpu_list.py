from dataclasses import is_dataclass, replace, dataclass
import torch


@dataclass
class DetachingCPUList(list):
    def append(self, item, detach_and_cpu=False):
        if detach_and_cpu:
            item = self._detach_recursive(item)
        super().append(item)

    def extend(self, iterable, detach_and_cpu=False):
        if detach_and_cpu:
            iterable = (self._detach_recursive(x) for x in iterable)
        super().extend(iterable)

    def insert(self, index, item, detach_and_cpu=False):
        if detach_and_cpu:
            item = self._detach_recursive(item)
        super().insert(index, item)

    def _detach_recursive(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: self._detach_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._detach_recursive(x) for x in obj)
        elif is_dataclass(obj):
            # Replace fields recursively (returns a new instance)
            return replace(obj, **{
                field.name: self._detach_recursive(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            })
        else:
            return obj