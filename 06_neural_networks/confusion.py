import numpy as np
confusion = [[765,  23,  16,  29,  20,   3,   4,  21,  74,  45, ],
             [16, 839,   3,  13,   4,   4,   3,   4,  22,  92, ],
             [113,  20, 350,  98, 147, 110,  43,  81,  19,  19, ],
             [33,  13,  25, 499,  75, 205,  21,  87,  19,  23, ],
             [15,   4,  34,  70, 595,  37,  16, 206,  19,   4, ],
             [13,   6,  17, 169,  49, 607,   5, 116,   5,  13, ],
             [7,  14,  21, 156, 148,  37, 578,  18,  15,   6, ],
             [17,   0,   6,  26,  41,  63,   2, 817,   3,  25, ],
             [80,  43,   2,  22,   5,  10,   0,   3, 787,  48, ],
             [38, 131,   1,  14,   4,   6,   2,  22,  34, 748, ]]

def format_matrix(matrix, environment="bmatrix", formatter=str):
    """Format a matrix using LaTeX syntax"""

    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception:
            raise TypeError("Could not convert to Numpy array")

    if len(shape := matrix.shape) == 1:
        matrix = matrix.reshape(1, shape[0])
    elif len(shape) > 2:
        raise ValueError("Array must be 2 dimensional")


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    body_lines = [classes[i] + " & " + " & ".join(map(formatter, row)) for i, row in enumerate(matrix)]

    body = "\\\\\n".join(body_lines)
    return f"""\\begin{{{environment}}}
    {"(empty) &" + " & ".join(classes)}\\\\
    {body}
    \\end{{{environment}}}"""
print(format_matrix(confusion))


from matplotlib import pyplot as plt

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
A = np.array(confusion)
ns = np.sum(A, axis=0)
A[np.identity(A.shape[0], np.bool8)] = 0
misclassifiactions = np.sum(A, axis=0)
# creating the bar plot
plt.bar(classes, misclassifiactions/ns, color='maroon',
        width=0.4)

plt.xlabel("Category")
plt.ylabel("Misclassification rate")
plt.title("Misclassification rate for each category")
plt.show()
