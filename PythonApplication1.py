
from typing import Any 
 
import math 
import sys 
 
sys.setrecursionlimit(1000000) 
 
 
def f(x): 
    return (1 - x[0]) ** 2 + 100 * ((x[1] - x[0] ** 2) ** 2) 
 
 
def dsk(x0, h, s): 
    xL = [x0[0] + h * s[0], x0[1] + h * s[1]] 
    if f(xL) >= f(x0): 
        h = -h 
    xL = [x0[0] + h * s[0], x0[1] + h * s[1]] 
    if f(xL) > f(x0): 
        x_k = [xL, x0, [x0[0] - h * s[0], x0[1] - h * s[1]]] 
        print(h) 
        x_k = dsk(x0, h * 0.5, s) 
        return x_k 
 
    x_k = [[x0[0], x0[1]], 
           [x0[0] + h * s[0], x0[1] + h * s[1]], 
           [x0[0] + 3 * h * s[0], x0[1] + 3 * h * s[1]]] 
 
    cur_x = [x0[0] + 3 * h * s[0], x0[1] + 3 * h * s[1]] 
 
    while f(cur_x) < f([x0[0] + h * s[0], x0[1] + h * s[1]]): 
        h = 2 * h 
        x_k[2] = x_k[1] 
        x_k[1] = cur_x 
        cur_x = [cur_x[0] + h * s[0], cur_x[1] + h * s[1]] 
        x_k[0] = cur_x 
    return x_k 
 
 
def DSC_Paul(x0, e, h, s): 
    x_k = dsk(x0, h, s) 
    # x_k = Sven(x0, s, 1) 
    print(x_k) 
    x1 = x_k[0] 
    x2 = x_k[1] 
    x3 = x_k[2] 
 
    sqx1 = math.sqrt(x1[0] ** 2 + x1[1] ** 2) 
    sqx2 = math.sqrt(x2[0] ** 2 + x2[1] ** 2) 
    sqx3 = math.sqrt(x3[0] ** 2 + x3[1] ** 2) 
 
    a1 = (f(x2) - f(x1)) / (sqx2 - sqx1) 
    a2 = (1 / (sqx3 - sqx1)) * ((f(x3) - f(x2)) / (sqx3 - sqx2) - (f(x2) - f(x1)) / (sqx2 - sqx1)) 
 
    x10 = [(x1[0] + x2[0]) / 2 - a1 / (2 * a2), (x1[1] + x2[1]) / 2 - a1 / (2 * a2)] 
 
    sqx10 = math.sqrt(x10[0] ** 2 + x10[1] ** 2) 
    diff_on_module_fx = abs(f(x2) - f(x10)) 
    diff_on_module_x = abs(sqx2 - sqx10) 
 
    s = [s[1], -s[0]] 
 
    # x_k = dsk(x0, h, s) 
    x_k = Sven(x0, s, 1) 
 
    x1 = x_k[0] 
    x2 = x_k[1] 
    x3 = x_k[2] 
 
    sqx1 = math.sqrt(x1[0] ** 2 + x1[1] ** 2) 
    sqx2 = math.sqrt(x2[0] ** 2 + x2[1] ** 2) 
    sqx3 = math.sqrt(x3[0] ** 2 + x3[1] ** 2) 
 
    a1 = (f(x2) - f(x1)) / (sqx2 - sqx1) 
    a2 = (1 / (sqx3 - sqx1)) * ((f(x3) - f(x2)) / (sqx3 - sqx2) - (f(x2) - f(x1)) / (sqx2 - sqx1)) 
 
    x11 = [(x1[0] + x2[0]) / 2 - a1 / (2 * a2), (x1[1] + x2[1]) / 2 - a1 / (2 * a2)] 
    print(f(x11), f(x10), 'x:', x11, x10) 
    if f(x10) >= f(x11): 
        sqx11 = math.sqrt(x11[0] ** 2 + x11[1] ** 2) 
        diff_on_module_fx = abs(f(x2) - f(x11)) 
        diff_on_module_x = abs(sqx2 - sqx11) 
        x10 = x11 
 
    if diff_on_module_fx < e or diff_on_module_x < e: 
        return x10 
    else: 
        lambda_i = DSC_Paul(x10, e, h, s) 
        return lambda_i 
 
 
def f_xy(x, y): 
    return (1 - x) ** 2 + 100 * ((y - x ** 2) ** 2) 
 
 
def Sven(sGold, xy, dxf): 
    dx = 0.01 
    la0 = 0 
    x0 = xy[0] 
    y0 = xy[1] 
 
    x = sGold[0] 
    y = sGold[1] 
 
    values_list = [f_xy(x0 + (la0 * x), y0 + (la0 * y))] 
    la_list = [la0] 
 
    if (f_xy(x0 + ((la0 - dx) * x), y0 + ((la0 - dx) * y)) > f_xy(x0 + (la0 * x), y0 + (la0 * y))) and ( 
            f_xy(x0 + (la0 * x), y0 + (la0 * y)) > f_xy(x0 + ((la0 + dx) * x), y0 + ((la0 + dx) * y))): 
        p = 1 
        values_list.append(f_xy(x0 + ((la0 + dx) * x), y0 + ((la0 + dx) * y))) 
        la_list.append(la0 + dx) 
    elif (f_xy(x0 + ((la0 - dx) * x), y0 + ((la0 - dx) * y)) < f_xy(x0 + (la0 * x), y0 + (la0 * y))) and ( 
            f_xy(x0 + (la0 * x), y0 + (la0 * y)) < f_xy(x0 + ((la0 + dx) * x), y0 + ((la0 + dx) * y))): 
        p = -1 
        values_list.append(f_xy(x0 + ((la0 - dx) * x), y0 + ((la0 - dx) * y))) 
        la_list.append(la0 - dx) 
    elif (f_xy(x0 + ((la0 - dx) * x), y0 + ((la0 - dx) * y)) >= f_xy(x0 + (la0 * x), y0 + (la0 * y))) and ( 
            f_xy(x0 + (la0 * x), y0 + (la0 * y)) <= f_xy(x0 + ((la0 + dx) * x), y0 + ((la0 + dx) * y))): 
        return [la0 - dx, la0 + dx] 
 
    i = 1 
 
    while values_list[i] < values_list[i - 1]: 
        la_i = la_list[i] + p * (2 ** i) * dx 
        la_list.append(la_i) 
 
        values_list.append(f_xy(x0 + (la_i * x), y0 + (la_i * y))) 
 
        i += 1 
 
    last = [la_list[i], (la_list[i] + la_list[i - 1]) / 2, la_list[i - 1], la_list[i - 2]] 
    last_e = [] 
 
    for la in last: 
        last_e.append(f_xy(x0 + (la * x), y0 + (la * y))) 
 
    if last_e[1] == min(last_e): 
        return sorted([last[2], last[0]]) 
 
    elif last_e[2] == min(last_e): 
        return sorted([last[3], last[1]]) 
 
 
def gold(xy, svenn, sGold): 
    x0 = xy[0] 
    y0 = xy[1] 
 
    x = sGold[0] 
    y = sGold[1] 
 
    cur_xk = svenn 
 
    L = cur_xk[1] - cur_xk[0] 
 
    la_1 = cur_xk[1] + 0.382 * L 
    la_2 = cur_xk[0] - 0.382 * L 
    f_la1 = f_xy(la_1 * x, la_1 * y) 
    f_la2 = f_xy(la_2 * x, la_2 * y) 
 
    while L > 0.01: 
        if f_la1 < f_la2: 
            f_la2 = f_la1 
            L = cur_xk[1] - cur_xk[0] 
            la_1 = cur_xk[0] + 0.382 * L 
            cur_xk = [cur_xk[0], la_2] 
            L = cur_xk[1] - cur_xk[0] 
            f_la1 = f_xy(la_1 * x,la_1 * y) 
 
        elif f_la1 > f_la2: 
            f_la1 = f_la2 
            L = cur_xk[1] - cur_xk[0] 
            la_2 = cur_xk[1] - 0.382 * L 
            cur_xk = [la_1, cur_xk[1]] 
            L = cur_xk[1] - cur_xk[0] 
            f_la2 = f_xy(la_2 * x, la_2 * y) 
    return (cur_xk[0]-cur_xk[1])/2 
 
 
def Rosenbrock(x_i, e, h, s, iter=1): 
    svenn = Sven(s, x_i, 0) 
    lambda_i = gold(x_i, svenn, s) 
    # lambda_i = DSC_Paul(x_i, e, h, s) 
    print("Лямбда", iter, " = ", lambda_i) 
    x_i1 = [x_i[0] + lambda_i * s[0], x_i[1] + lambda_i * s[1]] 
 
    x_i = x_i1 
 
    s = [-s[1], s[0]] 
 
    svenn = Sven(s, x_i1, 0) 
    lambda_i = gold(x_i1, svenn, s) 
    print("Лямбда", iter, " = ", lambda_i) 
    x_i1 = [x_i[0] + lambda_i * s[0], x_i[1] + lambda_i * s[1]] 
    diff_on_module_fx = abs(f(x_i1) - f(x_i)) / f(x_i1) 
 
    diff_on_module_x0 = 0 
    diff_on_module_x1 = 0 
 
    if x_i1[0] != 0: 
        diff_on_module_x0 = abs(x_i1[0] - x_i[0]) / x_i1[0] 
    if x_i1[1] != 0: 
        diff_on_module_x1 = abs(x_i1[1] - x_i[1]) / x_i1[1] 
 
    s = [x_i1[0] - x_i[0], -(x_i1[1] - x_i[1])] 
 
    if diff_on_module_fx < e and diff_on_module_x0 < e and diff_on_module_x1 < e: 
        print("Точка (x,y) =", x_i1, " min при e =", e, "f(", x_i1[0], ",", x_i1[1], ") = ", f(x_i1)) 
        exit() 
    else: 
        Rosenbrock(x_i1, e, h, s, iter + 1) 
 
 
x0 = [-0.5, 0] 
S = [1, 0] 
e = 0.01 
h = 0.1 
Rosenbrock(x0, e, h, S) 