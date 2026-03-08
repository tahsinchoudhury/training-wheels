import math

ITERATIONS = 1000

def underflow():
    print("[Underflow Issue]")
    p = 1e-5
    res = 1.0

    for i in range(ITERATIONS):
        res *= p
        print(f"Step: {i + 1}, result: {res}")

        if res == 0.0:
            print(f"Underflow at step: {i + 1}")
            break

def log():
    print("[Log Fix]")
    p = 1e-5
    logres = 0.0

    # logarithm sum
    for i in range(ITERATIONS):
        logres += math.log(p)

    # converting back
    res = math.exp(logres)
    print(f"logres: {logres}, res: {res}")

    # Takeaway: always compare the log-sums instead of exponentiating back

def exp_underflow():
    # problem:
    # suppose we want to compute log(exp(a) + exp(b)) where a, b are very small, negative numbers
    # exp(a) + exp(b) will be roughly zero, so this is a real problem because the log term will quickly become very large
    print("[Exponential Underflow]")
    a, b = -800, -800

    try:
        logsum = math.log(math.exp(a) + math.exp(b))
        print(f"log sum: {logsum}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Reason: exp(a) + exp(b) underflowed to 0")
        

def log_sum_exp():
    print("[Exponential Underflow]")
    a, b = -800, -800
    m = max(a, b)
    logsum = math.log(math.exp(a - m) + math.exp(b - m)) + m
    print(f"log sum: {logsum}")

if __name__ == "__main__":
    underflow()
    log()
    exp_underflow()
    log_sum_exp()