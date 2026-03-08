class MatrixOps:

    @staticmethod
    def tensor_multiply(arr1, arr2):
        batch_size = len(arr1)
        rows = len(arr1[0])
        cols = len(arr2[0][0])

        res = [
            [
                [0 for _ in range(cols)] for _ in range(rows)
            ] for _ in range(batch_size)
        ]

        for b in range(batch_size):
            for i in range(rows):
                for j in range(cols):
                    for k in range(len(arr1[b][0])):
                        res[b][i][j] += arr1[b][i][k] * arr2[b][k][j]

        return res

    @staticmethod
    def tensor_dot(arr1, arr2, dim):

        def get_shape(a):
            shape = []
            while isinstance(a, list):
                shape.append(len(a))
                if len(a) == 0:
                    break
                a = a[0]
            return shape

        def get_value(a, index):
            for i in index:
                a = a[i]
            return a

        def set_value(a, index, value):
            for i in index[:-1]:
                a = a[i]
            a[index[-1]] = value

        def make_tensor(shape):
            if not shape:
                return 0
            return [make_tensor(shape[1:]) for _ in range(shape[0])]

        def all_indices(shape):
            if not shape:
                yield ()
            else:
                for i in range(shape[0]):
                    for rest in all_indices(shape[1:]):
                        yield (i,) + rest

        shape = get_shape(arr1)

        if shape != get_shape(arr2):
            raise ValueError("Input arrays must have same shape")

        if dim < 0 or dim >= len(shape):
            raise ValueError("Invalid dimension")

        if len(shape) == 1:
            return sum(a * b for a, b in zip(arr1, arr2))

        out_shape = shape[:dim] + shape[dim + 1:]

        if not out_shape:
            total = 0
            for i in range(shape[dim]):
                total += arr1[i] * arr2[i]
            return total

        result = make_tensor(out_shape)

        for out_index in all_indices(out_shape):
            total = 0

            for k in range(shape[dim]):
                full_index = out_index[:dim] + (k,) + out_index[dim:]
                total += (
                    get_value(arr1, full_index)
                    * get_value(arr2, full_index)
                )

            set_value(result, out_index, total)

        return result