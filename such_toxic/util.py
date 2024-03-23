from typing import List, Union


def shape(data: List) -> List[int]:
    if not data:
        return [0]

    if not isinstance(data, list):
        # If it's not a list, assume it's a scalar and return an empty tuple
        return []

    # Get the shape of the first element to check for nested structures
    first_element = data[0]
    if isinstance(first_element, list):
        # If the first element is a list, get the nested shape and prepend the outer list length
        return [len(data)] + shape(first_element)
    else:
        # If the elements are not lists, return a tuple with the length of the outer list
        return [len(data)]


def unsqueeze(data: List, axis: int = 0) -> List:
    if axis < 0 or axis > len(shape(data)):
        raise ValueError("Invalid axis value")
    if axis == 0:
        return [data]  # Unsqueeze at the beginning (axis 0)
    else:
        # Simulate unsqueeze along other axes by nesting lists
        return [unsqueeze(inner_list, axis - 1) for inner_list in data]


def expand(data: List, size: List[int]) -> List:
    data_shape = shape(data)

    if data_shape == [1] and size == [1]:
        return data
    if data_shape == [1] and len(size) == 1:
        return data * size[0]
    if len(size) != len(data_shape):
        raise ValueError("Size must have the same length as the data shape")
    if size[0] != len(data):
        raise ValueError("First dimension of size must match the length of the data")

    return [expand(inner_list, size[1:]) for inner_list in data]


def mat_mul(a: List, b: List) -> List:
    if shape(a) != shape(b):
        raise ValueError("Shapes of both lists must match")

    if isinstance(a[0], list):
        return [mat_mul(a[i], b[i]) for i in range(len(a))]
    else:
        return [a[i] * b[i] for i in range(len(a))]


def transpose(data: List) -> List:
    if not data:
        return data

    if not isinstance(data[0], list):
        return [[data[i]] for i in range(len(data))]

    return [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]


def mat_sum(a: List, dim: int = 0) -> List:
    shape_data = shape(a)
    print("Shape data: ", shape_data)
    print("Dim: ", dim)

    # we want to sum across the given dimension
    if dim < 0 or dim >= len(shape_data):
        raise ValueError("Invalid dimension value")

    if len(shape_data) == 1:
        # 1d list, sum all elements
        return sum(a)

    # we have a multi-dimensional list
    # depending on the dim value, we need to go deeper until we reach the desired dimension
    # then begin summing the elements
    # dim 0 means we sum accross every sublist

    if dim == 0:
        return sum([mat_sum(inner_list, dim) for inner_list in a])

    a = transpose(a)
    return [mat_sum(inner_list, dim - 1) for inner_list in a]


def clamp(data: List, min_val: float, max_val: float) -> List:
    if isinstance(data[0], list):
        return [clamp(sublist, min_val, max_val) for sublist in data]
    else:
        return [min(max_val, max(min_val, val)) for val in data]
