# shape: List[], e.g. shape=[2, 3, 4] creates an nd array of shape (2, 3, 4) 
def zeros(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    return [zeros(shape[1:]) for _ in range(shape[0])]

def get_shape(arr):
    shape = []
    while isinstance(arr, list):
        shape.append(len(arr))
        if len(arr) == 0:
            break
        arr = arr[0]
    return shape
