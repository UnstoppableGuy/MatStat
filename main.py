import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
from scipy import stats


def csv_dict_writer(path):
    csv_dict = []
    with open(path, encoding='utf-8') as r_file:
        file_reader = csv.reader(r_file, delimiter=",")
        for row in file_reader:
            csv_dict = list(map(float, row))
    return csv_dict


def get_mu(data):
    s = 0
    for val in data:
        s += val
    s /= len(data)
    return s


def get_disp(mu, data):
    disp = 0
    for item in data:
        disp += ((item - mu)**2)
    disp /= (len(data)-1)
    return disp


def plotting_histogram(data):
    domain = np.arange(min(data), max(data))
    n, autobins, patches = plt.hist(data, density=1, bins=len(domain),
                                    color='green')  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    n = n*(autobins[1]-autobins[0])
    return n


def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def normal_distribution(data):
    mu = get_mu(data)
    disp = get_disp(mu, data)
    sigma = np.sqrt(disp)
    #domain = np.linspace(min(data), max(data), 1000)
    domain = np.arange(min(data), max(data))
    y_deduced = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                 np.exp(-0.5 * (1 / sigma * (domain - mu))**2))
    plt.plot(domain, y_deduced, '-', color='red', label='normal')
    return y_deduced


def exp_distribution(data):
    lambda_mu = 1 / get_mu(data)
    lambda_disp = 1 / np.sqrt(get_disp(get_mu(data), data))
    #domain = np.linspace(0, max(data), 1000)
    domain = np.arange(0, max(data))
    y_deduced = lambda_disp * np.exp(-lambda_disp * domain)
    plt.plot(domain+1, y_deduced, '--', color='cyan', label='exponential')
    return y_deduced


def geometric_distribution(data):
    p = 1 / get_mu(data)
    if min(data) < 0:
        x = np.arange(0, int(max(data)))
    else:
        x = np.arange(int(min(data)), int(max(data)))
    y_deduced = []
    for item in x:
        y_deduced.append(((1-p)**item)*p)
    plt.plot(x, y_deduced, color='magenta', label='geometric')
    return y_deduced


def poisson_distribution(data):
    mu = get_mu(data)
    if min(data) < 0:
        x = np.arange(0, int(max(data)))
    else:
        x = np.arange(int(min(data)), int(max(data)))
    y_deduced = []
    for item in x:
        y_deduced.append(np.exp(-mu) * mu**item / math.factorial(item))
    plt.plot(x, y_deduced, '-', color='pink', label='poisson')
    return y_deduced


def binomial_distribution(data):
    mu = get_mu(data)
    value = int(max(data))
    p = mu / value
    domain = np.linspace(min(data), max(data), value)
    y_deduced = []
    for i in range(value):
        y_deduced.append(binom(value, i) * p**i * (1-p)**(value - i))

    plt.plot(domain, y_deduced, '-', color='orange', label='binomial')
    return y_deduced


def unifrom_distribution(data):
    a = min(data)
    b = max(data)
    points_num = 100
    y = 1 / (b-a)
    domain = np.linspace(a, b, points_num)
    plt.plot(domain, [y]*points_num, '--', color='black', label='uniform')
    return [y]*points_num


def t_test(data, left, right):
    mo = get_mu(data)
    sigma = np.sqrt(get_disp(mo, data))
    t = (mo - left) / sigma
    result = stats.t.cdf(t, len(data)-1)
    print(f"p-value:{result}\n")


def chi_square(emperal, teoretic, text):
    result = 0
    for item in range(len(emperal)):
        result += (emperal[item]-teoretic[item])**2/teoretic[item]
    print(f"Кси квадрат для {text} распределения: {result}")
    #print(stats.chisquare(res), "\n")


'''
def chi2_contingency(data1, data2):
    chi2, prob, df, expected = stats.chi2_contingency([[data1], [data2]])
    output = "test Statistics: {}\ndegrees of freedom: {}\np-value: {}\n"
    print(output.format(chi2, df, prob))

'''


"""def test(data1, data2):
    stat, p, dof, expected = stats.chi2_contingency([data1,data2])
    print('dof=%d' % dof)
    #print(expected)
    # interpret test-statistic'''
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' %
          (prob, critical, stat))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    # interpret p-value'''
    prob = 0.95
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')"""


def anaylize_data(data, name):
    mo = get_mu(data)
    disp = np.sqrt(get_disp(mo, data))
    interval_estimation_left = mo - np.sqrt(get_disp(mo, data))
    interval_estimation_right = mo + np.sqrt(get_disp(mo, data))
    plt.vlines(mo, ymin=0, ymax=0.3, linestyles='-',
               color='blue', label='point estimate')
    plt.vlines(interval_estimation_left, ymin=0, ymax=0.3,
               linestyles='dotted', color='blue', label='interval estimate')
    plt.vlines(interval_estimation_right, ymin=0,
               ymax=0.3, linestyles='dotted', color='blue')
    print(
        f"Точечная оценка:{get_mu(data)}\nИнтервальная оценка [{interval_estimation_left}|{interval_estimation_right}]")
    t_test(data, interval_estimation_left, interval_estimation_right)

    '''1)normal 2)exponential 3)geometric'''
    n = plotting_histogram(data)
    val = []
    names = ['normal', 'exponential', 'binomial',
             'uniform', 'poisson', 'geomtric']
    val.append(normal_distribution(data))  # 0
    val.append(exp_distribution(data))  # 1
    val.append(binomial_distribution(data))  # 2
    val.append(unifrom_distribution(data))  # 3
    val.append(poisson_distribution(data))  # 4
    val.append(geometric_distribution(data))  # 5
    #for item in range(len(val)):
    #    chi_square(n, val[item], text=names[item])
    #chi_square(n, val[0])
    # test(n,val[0])
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.show()
    plt.cla()


if __name__ == '__main__':
    data1 = csv_dict_writer('set_1.csv')
    data2 = csv_dict_writer('set_2.csv')
    data3 = csv_dict_writer('set_3.csv')
    anaylize_data(data1, '1')
    anaylize_data(data2, '2')
    anaylize_data(data3, '3')
