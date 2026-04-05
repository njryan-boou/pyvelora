def _is_scalar(x):
    return isinstance(x, (int, float))


def _to_list(y0):
    if _is_scalar(y0):
        return [y0]
    return list(y0)


def _from_list(y0, arr):
    if _is_scalar(y0):
        return arr[0]
    return arr